// loader_hf.cpp — Direct HuggingFace safetensors model loader
// Reads config.json + model.safetensors + tokenizer.json from HF dir,
// quantizes on-the-fly, outputs the same TensorMap interface as loader.cpp.
#include "core/config.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "core/json.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>

using json = nlohmann::json;

// ── Safetensors header parsing ──────────────────────────────────

struct SafeTensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset;  // byte offset in the file
    uint64_t size;    // byte size
};

static bool parse_safetensors(const std::string& path,
                               std::vector<SafeTensorEntry>& entries,
                               uint64_t& data_start) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    uint64_t hdr_len;
    f.read((char*)&hdr_len, 8);
    if (hdr_len > 100 * 1024 * 1024) {  // sanity: max 100MB header
        fprintf(stderr, "safetensors header too large: %llu\n", (unsigned long long)hdr_len);
        return false;
    }

    std::string hdr_json((size_t)hdr_len, '\0');
    f.read(&hdr_json[0], hdr_len);
    data_start = 8 + hdr_len;

    json hdr;
    try { hdr = json::parse(hdr_json); }
    catch (...) { fprintf(stderr, "safetensors JSON parse error\n"); return false; }

    for (auto& [key, val] : hdr.items()) {
        if (key == "__metadata__") continue;
        SafeTensorEntry e;
        e.name = key;
        e.dtype = val["dtype"].get<std::string>();
        e.shape = val["shape"].get<std::vector<int64_t>>();
        auto offs = val["data_offsets"].get<std::vector<uint64_t>>();
        e.offset = data_start + offs[0];
        e.size = offs[1] - offs[0];
        entries.push_back(e);
    }
    return true;
}

// ── Dtype conversion helpers ────────────────────────────────────

static int dtype_to_elemsize(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16") return 2;
    if (dtype == "I64") return 8;
    return 0;
}

static void convert_f16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        // FP16 → FP32 conversion
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32;
        if (exp == 0) {
            f32 = (sign << 31) | (0x7F - 15 + 1) << 23 | mant << 13;
        } else if (exp == 31) {
            f32 = (sign << 31) | 0xFF << 23 | (mant ? 0x7FFFFF : 0);
        } else {
            f32 = (sign << 31) | (exp + 127 - 15) << 23 | mant << 13;
        }
        memcpy(dst + i, &f32, 4);
    }
}

static void convert_bf16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(dst + i, &bits, 4);
    }
}

// ── Architecture detection ──────────────────────────────────────

struct HFArchConfig {
    std::string rina_name;
    int n_layers;
    int dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int d_h_r;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    bool weight_tying;
};

static HFArchConfig parse_hf_config(const std::string& cfg_path) {
    std::ifstream f(cfg_path);
    json cfg; f >> cfg;
    HFArchConfig r{};
    r.dim = cfg["hidden_size"];
    r.n_layers = cfg["num_hidden_layers"];
    r.n_heads = cfg["num_attention_heads"];
    r.n_kv_heads = cfg.value("num_key_value_heads", r.n_heads);
    r.head_dim = cfg.value("head_dim", r.dim / r.n_heads);
    r.d_h_r = r.head_dim;
    r.vocab_size = cfg["vocab_size"];
    r.max_seq_len = std::min(cfg.value("max_position_embeddings", 512), 2048);
    r.rope_theta = cfg.value("rope_theta", 10000.0f);
    r.weight_tying = cfg.value("tie_word_embeddings", true);
    // Name from model_type
    std::string mt = cfg.value("model_type", "llama");
    if (mt == "llama") r.rina_name = "llama";
    else if (mt == "mistral") r.rina_name = "mistral";
    else if (mt == "qwen2") r.rina_name = "qwen2";
    else r.rina_name = mt;
    return r;
}

// ── Model layer type mapping ────────────────────────────────────

static bool is_layer_weight(const std::string& name) {
    return name.rfind("model.layers.", 0) == 0;
}
static int layer_index(const std::string& name) {
    if (name.rfind("model.layers.", 0) != 0) return -1;
    auto p = name.find('.', 13); // after "model.layers."
    if (p == std::string::npos) return -1;
    return std::stoi(name.substr(13, p - 13));
}
static std::string layer_local_name(const std::string& name) {
    auto p = name.find('.', 13);
    if (p == std::string::npos) return "";
    auto p2 = name.find('.', p + 1);
    if (p2 == std::string::npos) return name.substr(p + 1);
    return name.substr(p + 1);
}

static std::string hf_to_rinn_name(const std::string& hf_name, int l) {
    // Global weights (not per-layer)
    if (hf_name == "model.embed_tokens.weight") return "transformer.wte.weight";
    if (hf_name == "model.norm.weight") return "transformer.ln_f.weight";
    if (hf_name.rfind("model.layers.", 0) != 0) return "";
    // Per-layer weights
    auto local = layer_local_name(hf_name);
    if (local == "input_layernorm.weight")
        return "transformer.h." + std::to_string(l) + ".ln1.weight";
    if (local == "post_attention_layernorm.weight")
        return "transformer.h." + std::to_string(l) + ".ln2.weight";
    if (local == "self_attn.q_proj.weight")
        return "transformer.h." + std::to_string(l) + ".attn.w_q.weight";
    if (local == "self_attn.k_proj.weight")
        return "transformer.h." + std::to_string(l) + ".attn.w_k.weight";
    if (local == "self_attn.v_proj.weight")
        return "transformer.h." + std::to_string(l) + ".attn.w_v.weight";
    if (local == "self_attn.o_proj.weight")
        return "transformer.h." + std::to_string(l) + ".attn.w_o.weight";
    if (local == "mlp.gate_proj.weight")
        return "transformer.h." + std::to_string(l) + ".mlp.w1.weight";
    if (local == "mlp.up_proj.weight")
        return "transformer.h." + std::to_string(l) + ".mlp.w3.weight";
    if (local == "mlp.down_proj.weight")
        return "transformer.h." + std::to_string(l) + ".mlp.w2.weight";
    return "";
}

static bool should_quant(const std::string& rinn_name) {
    if (rinn_name.find("wte") != std::string::npos) return false;
    if (rinn_name.find("ln1") != std::string::npos) return false;
    if (rinn_name.find("ln2") != std::string::npos) return false;
    if (rinn_name.find("ln_f") != std::string::npos) return false;
    if (rinn_name.find("lm_head") != std::string::npos) return false;
    if (rinn_name.find("rope") != std::string::npos) return false;
    return rinn_name.find(".weight") != std::string::npos;
}

// ── Quantize on-the-fly ─────────────────────────────────────────

static void quantize_q4_0(const float* src, std::vector<uint8_t>& dst, int n) {
    int nb = (n + 31) / 32;
    dst.resize(nb * 18);
    for (int b = 0; b < nb; b++) {
        int start = b * 32;
        int end = std::min(start + 32, n);
        // Find max absolute
        float amax = 0.0f;
        for (int i = start; i < end; i++) amax = std::max(amax, fabsf(src[i]));
        float scale = amax / 7.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        // Store half scale
        uint16_t scale_h;
        float scale_f = scale;
        uint32_t* p = (uint32_t*)&scale_f;
        uint32_t sign = (*p >> 31) & 1;
        int32_t exp = ((*p >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (*p >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; mant = 0; }
        else if (exp >= 31) { exp = 31; mant = 0x3FF; }
        scale_h = (sign << 15) | (exp << 10) | mant;
        memcpy(&dst[b * 18], &scale_h, 2);
        // Quantize
        for (int i = start; i < end; i++) {
            float qf = src[i] / scale;
            int qi = std::max(-7, std::min(7, (int)roundf(qf)));
            int qu = qi + 7;
            int idx = i - start;
            if (idx % 2 == 0)
                dst[b * 18 + 2 + idx / 2] = (dst[b * 18 + 2 + idx / 2] & 0xF0) | (qu & 0x0F);
            else
                dst[b * 18 + 2 + idx / 2] = (dst[b * 18 + 2 + idx / 2] & 0x0F) | ((qu & 0x0F) << 4);
        }
    }
}

// ── Main HF loader ──────────────────────────────────────────────

bool load_hf_model(const char* dir_path, ModelConfig& cfg, TensorMap& tensors,
                   int quant_bits = 4) {
    std::string base = dir_path;
    // Read config
    auto hf = parse_hf_config(base + "/config.json");
    // Build RINN config
    cfg.name = hf.rina_name + "-" + std::to_string(hf.dim / 1000) + "dim";
    cfg.dim = hf.dim;
    cfg.n_layers = hf.n_layers;
    cfg.n_heads = hf.n_heads;
    cfg.n_kv_heads = hf.n_kv_heads;
    cfg.head_dim = hf.head_dim;
    cfg.d_c = 0;
    cfg.d_h_r = hf.d_h_r;
    cfg.vocab_size = hf.vocab_size;
    cfg.max_seq_len = hf.max_seq_len;
    cfg.ssm_steps = 0;
    cfg.weight_tying = hf.weight_tying;
    cfg.layers.clear();
    for (int i = 0; i < hf.n_layers; i++)
        cfg.layers.push_back({"standard_attention", "layer_" + std::to_string(i), 1, {}});

    // Find safetensors file
    std::string sf_path;
    for (auto& p : {base + "/model.safetensors", base + "/model-00001-of-00002.safetensors",
                     base + "/consolidated.safetensors"}) {
        struct stat st;
        if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            sf_path = p; break;
        }
    }
    if (sf_path.empty()) {
        // Globsafetensors
        for (auto& p : std::filesystem::directory_iterator(base)) {
            auto s = p.path().string();
            if (s.size() > 12 && s.substr(s.size()-12) == ".safetensors") {
                sf_path = s; break;
            }
        }
    }
    if (sf_path.empty()) { fprintf(stderr,"No safetensors found in %s\n", dir_path); return false; }

    // Parse safetensors header
    std::vector<SafeTensorEntry> entries;
    uint64_t data_start;
    if (!parse_safetensors(sf_path, entries, data_start)) return false;
    fprintf(stderr, "  safetensors: %zu tensors, quant=%d\n", entries.size(), quant_bits);

    // Open file for tensor read
    FILE* sf = fopen(sf_path.c_str(), "rb");
    if (!sf) { fprintf(stderr,"Cannot open %s\n", sf_path.c_str()); return false; }

    // Generate RoPE tables for each layer
    fprintf(stderr, "  generating RoPE tables...\n");
    int max_seq = hf.max_seq_len;
    int d_h_r = hf.d_h_r;
    int half = d_h_r / 2;
    std::vector<float> cos_tbl(max_seq * half);
    std::vector<float> sin_tbl(max_seq * half);
    float theta = hf.rope_theta;
    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)d_h_r);
        for (int t = 0; t < max_seq; t++) {
            float val = (float)t * inv_freq;
            cos_tbl[t * half + i] = cosf(val);
            sin_tbl[t * half + i] = sinf(val);
        }
    }

    // Process tensors
    int n_quantized = 0, n_fp32 = 0;
    for (auto& entry : entries) {
        int l = layer_index(entry.name);
        std::string rinn_name;
        if (l >= 0) rinn_name = hf_to_rinn_name(entry.name, l);
        else rinn_name = hf_to_rinn_name(entry.name, 0);

        if (rinn_name.empty()) continue;  // skip biases, etc

        // Read raw data
        int esize = dtype_to_elemsize(entry.dtype);
        int n_elems = 1;
        for (auto s : entry.shape) n_elems *= (int)s;
        std::vector<float> f32_buf(n_elems);
        fseek(sf, entry.offset, SEEK_SET);

        if (entry.dtype == "F32") {
            fread(f32_buf.data(), 4, n_elems, sf);
        } else if (entry.dtype == "F16") {
            std::vector<uint16_t> hbuf(n_elems);
            fread(hbuf.data(), 2, n_elems, sf);
            convert_f16_to_f32(hbuf.data(), f32_buf.data(), n_elems);
        } else if (entry.dtype == "BF16") {
            std::vector<uint16_t> hbuf(n_elems);
            fread(hbuf.data(), 2, n_elems, sf);
            convert_bf16_to_f32(hbuf.data(), f32_buf.data(), n_elems);
        } else {
            fprintf(stderr, "  unsupported dtype %s for %s\n", entry.dtype.c_str(), rinn_name.c_str());
            continue;
        }

        bool q = should_quant(rinn_name) && quant_bits == 4;
        WeightTensor wt;
        wt.n_dim = (int)entry.shape.size();
        for (int i = 0; i < wt.n_dim; i++) wt.shape[i] = (int)entry.shape[i];
        wt.n_elems = n_elems;

        if (q) {
            std::vector<uint8_t> qpacked;
            quantize_q4_0(f32_buf.data(), qpacked, n_elems);
            wt.quant_type = QuantType::Q4_0;
            size_t bytes = qpacked.size();
            cudaMalloc(&wt.data, bytes);
            cudaMemcpy(wt.data, qpacked.data(), bytes, cudaMemcpyHostToDevice);
            n_quantized++;
        } else {
            wt.quant_type = QuantType::FP32;
            size_t bytes = n_elems * sizeof(float);
            cudaMalloc(&wt.data, bytes);
            cudaMemcpy(wt.data, f32_buf.data(), bytes, cudaMemcpyHostToDevice);
            n_fp32++;
        }
        tensors.add(rinn_name, std::move(wt));
    }

    // Add RoPE tables as fp32 tensors for each layer
    for (int l = 0; l < hf.n_layers; l++) {
        auto add_rope = [&](const std::string& name, float* data) {
            WeightTensor wt;
            wt.quant_type = QuantType::FP32;
            wt.n_dim = 2;
            wt.shape[0] = max_seq; wt.shape[1] = half; wt.shape[2] = 0; wt.shape[3] = 0;
            wt.n_elems = max_seq * half;
            size_t bytes = wt.n_elems * sizeof(float);
            cudaMalloc(&wt.data, bytes);
            cudaMemcpy(wt.data, data, bytes, cudaMemcpyHostToDevice);
            tensors.add(name, std::move(wt));
        };
        std::string p = "transformer.h." + std::to_string(l) + ".attn.";
        add_rope(p + "rope_q.cos", cos_tbl.data());
        add_rope(p + "rope_q.sin", sin_tbl.data());
        add_rope(p + "rope.cos", cos_tbl.data());
        add_rope(p + "rope.sin", sin_tbl.data());
    }

    fclose(sf);
    fprintf(stderr, "  loaded: %d fp32 + %d q4_0 tensors\n", n_fp32, n_quantized);
    return true;
}
