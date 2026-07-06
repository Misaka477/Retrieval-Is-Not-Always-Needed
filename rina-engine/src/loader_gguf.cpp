// loader_gguf.cpp — Direct GGUF model loader
// Reads GGUF format models (llama.cpp ecosystem), dequantizes on-the-fly,
// outputs the same TensorMap interface as loader.cpp.
#include "core/config.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "loader/gguf_reader.h"
#include "loader/loader_base.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <sys/stat.h>

// ── HF→RINN name mapping (reuses logic from loader_hf) ─────────

static bool is_layer_weight(const std::string& name) {
    return name.rfind("blk.", 0) == 0;
}

static int layer_index(const std::string& gguf_name) {
    if (gguf_name.rfind("blk.", 0) != 0) return -1;
    auto p = gguf_name.find('.', 4);
    return std::stoi(gguf_name.substr(4, p - 4));
}

static std::string gguf_to_rinn_name(const std::string& gguf_name, int l) {
    if (gguf_name == "token_embd.weight")    return "transformer.wte.weight";
    if (gguf_name == "output_norm.weight")   return "transformer.ln_f.weight";
    if (gguf_name == "output.weight")        return "lm_head.weight";  // not tied for MLA

    if (!is_layer_weight(gguf_name)) return "";
    int layer = layer_index(gguf_name);
    if (layer < 0) return "";

    std::string local = gguf_name.substr(gguf_name.find('.', 4) + 1);
    std::string prefix = "transformer.h." + std::to_string(layer) + ".";

    if (local == "attn_norm.weight")        return prefix + "ln1.weight";
    if (local == "ffn_norm.weight")         return prefix + "ln2.weight";
    if (local == "attn_q.weight")           return prefix + "attn.w_q.weight";
    if (local == "attn_k.weight")           return prefix + "attn.w_k.weight";
    if (local == "attn_v.weight")           return prefix + "attn.w_v.weight";
    if (local == "attn_output.weight")      return prefix + "attn.w_o.weight";
    if (local == "ffn_gate.weight")         return prefix + "mlp.w1.weight";
    if (local == "ffn_up.weight")           return prefix + "mlp.w3.weight";
    if (local == "ffn_down.weight")         return prefix + "mlp.w2.weight";

    if (local == "attn_kv_a_mqa.weight")    return prefix + "attn.w_kv_a.weight";
    if (local == "attn_kv_a_norm.weight")   return prefix + "attn.k_norm.weight";
    if (local == "attn_kv_b.weight")        return prefix + "attn.w_kv_b.weight";
    if (local == "ffn_gate_inp.weight")     return prefix + "mlp.gate_inp.weight";
    if (local == "ffn_gate_exps.weight")    return prefix + "mlp.gate_exps.weight";
    if (local == "ffn_up_exps.weight")      return prefix + "mlp.up_exps.weight";
    if (local == "ffn_down_exps.weight")    return prefix + "mlp.down_exps.weight";
    if (local == "ffn_gate_shexp.weight")   return prefix + "mlp.gate_shexp.weight";
    if (local == "ffn_up_shexp.weight")     return prefix + "mlp.up_shexp.weight";
    if (local == "ffn_down_shexp.weight")   return prefix + "mlp.down_shexp.weight";

    return "";
}

// ── Main GGUF loader ──────────────────────────────────────────

bool load_gguf_model(const char* path, ModelConfig& cfg, TensorMap& tensors, int max_layers) {
    GGUFMetadata meta;
    if (!read_gguf_header(path, meta))
        return false;

    // Detect architecture and build config
    const ArchLoader* arch = detect_arch(meta);
    if (!arch) {
        fprintf(stderr, "ERROR: unknown architecture '%s'\n", meta.arch_string.c_str());
        return false;
    }
    if (!arch->build_config(cfg, meta)) {
        fprintf(stderr, "ERROR: failed to build config for '%s'\n", arch->name.c_str());
        return false;
    }

    // Open file for tensor reads
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    // Generate RoPE tables
    int max_seq = cfg.max_seq_len > 0 ? cfg.max_seq_len : (meta.max_seq_len > 0 ? meta.max_seq_len : 2048);
    int half = cfg.head_dim / 2;
    std::vector<float> cos_tbl(max_seq * half);
    std::vector<float> sin_tbl(max_seq * half);
    generate_rope_tables(max_seq, cfg.head_dim, cos_tbl, sin_tbl,
                         meta.rope_freq_base, meta.rope_scaling_factor);

    // Single buffer allocation: first pass calculates total size
    size_t total_weight_bytes = 0;
    for (auto& gt : meta.tensors) {
        int64_t n_elems = 1;
        for (auto s : gt.shape) n_elems *= s;
        int blk_sz = ggml_block_size(gt.ggml_type);
        int type_sz = ggml_type_size(gt.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t sz = (size_t)n_blocks * type_sz;
        total_weight_bytes = (total_weight_bytes + 255) & ~255;
        total_weight_bytes += sz;
    }
    // Additional space for RoPE tables and dequant_tmp
    size_t rope_bytes = (size_t)max_seq * half * sizeof(float) * 2;  // cos + sin
    size_t dequant_tmp_bytes = (size_t)max_seq * cfg.dim * sizeof(float);
    size_t total_alloc = total_weight_bytes + rope_bytes + dequant_tmp_bytes;

    void* weight_buf;
    cudaMalloc(&weight_buf, total_alloc);
    if (!weight_buf) {
        fprintf(stderr, "ERROR: weight buffer OOM (%zu MB)\n", total_alloc / 1048576);
        fclose(f);
        return false;
    }
    tensors.shared_buffer = weight_buf;

    // Precompute offsets for all tensors
    std::vector<size_t> tensor_offsets(meta.tensors.size());
    size_t running_offset = 0;
    for (int i = 0; i < (int)meta.tensors.size(); i++) {
        auto& gt = meta.tensors[i];
        int64_t n_elems = 1;
        for (auto s : gt.shape) n_elems *= s;
        int blk_sz = ggml_block_size(gt.ggml_type);
        int type_sz = ggml_type_size(gt.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t sz = (size_t)n_blocks * type_sz;
        running_offset = (running_offset + 255) & ~255;
        tensor_offsets[i] = running_offset;
        running_offset += sz;
    }

    // Second pass: read and upload each tensor into the shared buffer
    int n_tensors_loaded = 0;
    for (int ti = 0; ti < (int)meta.tensors.size(); ti++) {
        auto& gt = meta.tensors[ti];
        size_t buf_offset = tensor_offsets[ti];

        std::string rinn_name = gguf_to_rinn_name(gt.name, 0);
        if (rinn_name.empty()) continue;
        if (max_layers > 0 && is_layer_weight(gt.name)) {
            int li = layer_index(gt.name);
            if (li >= max_layers) continue;
        }

        int64_t n_elems = 1;
        for (auto s : gt.shape) n_elems *= s;

        int blk_sz = ggml_block_size(gt.ggml_type);
        int type_sz = ggml_type_size(gt.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t raw_size = (size_t)n_blocks * type_sz;

        // GGML quant types: upload as quantized blocks for GPU dequant+matmul
        bool is_ggml_quant = (gt.ggml_type == GGML_TYPE_Q2_K ||
                              gt.ggml_type == GGML_TYPE_Q3_K ||
                              gt.ggml_type == GGML_TYPE_Q4_K ||
                              gt.ggml_type == GGML_TYPE_Q5_K ||
                              gt.ggml_type == GGML_TYPE_Q6_K ||
                              gt.ggml_type == GGML_TYPE_IQ3_XXS ||
                              gt.ggml_type == GGML_TYPE_IQ3_S ||
                              gt.ggml_type == GGML_TYPE_IQ4_NL ||
                              gt.ggml_type == GGML_TYPE_IQ4_XS);
        if (is_ggml_quant) {

            std::vector<uint8_t> raw(raw_size);
            fseek(f, gt.offset, SEEK_SET);
            fread(raw.data(), 1, raw_size, f);

            WeightTensor wt;
            wt.n_dim = (int)gt.shape.size();
            if (wt.n_dim >= 2 && rinn_name.find(".weight") != std::string::npos) {
                wt.shape[1] = (int)gt.shape[0];
                wt.shape[0] = (int)gt.shape[1];
                for (int i = 2; i < wt.n_dim; i++) wt.shape[i] = (int)gt.shape[i];
            } else {
                for (int i = 0; i < wt.n_dim; i++) wt.shape[i] = (int)gt.shape[i];
            }
            wt.n_elems = n_elems;
            switch (gt.ggml_type) {
                case GGML_TYPE_Q2_K:  wt.quant_type = QuantType::GGML_Q2_K; break;
                case GGML_TYPE_Q3_K:  wt.quant_type = QuantType::GGML_Q3_K; break;
                case GGML_TYPE_Q4_K:  wt.quant_type = QuantType::GGML_Q4_K; break;
                case GGML_TYPE_Q5_K:  wt.quant_type = QuantType::GGML_Q5_K; break;
                case GGML_TYPE_Q6_K:  wt.quant_type = QuantType::GGML_Q6_K; break;
                case GGML_TYPE_IQ3_XXS: wt.quant_type = QuantType::GGML_IQ3_XXS; break;
                case GGML_TYPE_IQ3_S: wt.quant_type = QuantType::GGML_IQ3_S; break;
                case GGML_TYPE_IQ4_NL: wt.quant_type = QuantType::GGML_IQ4_NL; break;
                case GGML_TYPE_IQ4_XS: wt.quant_type = QuantType::GGML_IQ4_XS; break;
            }
            wt.data = (uint8_t*)weight_buf + buf_offset;
            wt.owned = false;
            cudaMemcpy(wt.data, raw.data(), raw_size, cudaMemcpyHostToDevice);
            tensors.add(rinn_name, std::move(wt));
            n_tensors_loaded++;
            continue;
        }

        std::vector<uint8_t> raw(raw_size);
        fseek(f, gt.offset, SEEK_SET);
        fread(raw.data(), 1, raw_size, f);

        std::vector<float> f32_buf(n_elems);
        switch (gt.ggml_type) {
            case GGML_TYPE_F32:
                memcpy(f32_buf.data(), raw.data(), n_elems * 4);
                break;
            case GGML_TYPE_F16:
                dequant_block_f16(raw.data(), f32_buf.data(), n_elems);
                break;
            case GGML_TYPE_Q4_0:
                dequant_q4_0(raw.data(), f32_buf.data(), n_elems);
                break;
            case GGML_TYPE_Q6_K:
                dequant_q6_K(raw.data(), f32_buf.data(), n_elems);
                break;
            default:
                fprintf(stderr, "  unsupported ggml_type=%d for %s\n",
                        gt.ggml_type, gt.name.c_str());
                memset(f32_buf.data(), 0, n_elems * sizeof(float));
                break;
        }

        WeightTensor wt;
        wt.n_dim = (int)gt.shape.size();
        if (wt.n_dim >= 2 && rinn_name.find(".weight") != std::string::npos) {
            wt.shape[1] = (int)gt.shape[0];
            wt.shape[0] = (int)gt.shape[1];
            for (int i = 2; i < wt.n_dim; i++) wt.shape[i] = (int)gt.shape[i];
        } else {
            for (int i = 0; i < wt.n_dim; i++) wt.shape[i] = (int)gt.shape[i];
        }
        wt.n_elems = n_elems;
        wt.quant_type = QuantType::FP32;
        size_t bytes = n_elems * sizeof(float);
        wt.data = (uint8_t*)weight_buf + buf_offset;
        wt.owned = false;
        cudaMemcpy(wt.data, f32_buf.data(), bytes, cudaMemcpyHostToDevice);
        tensors.add(rinn_name, std::move(wt));
        n_tensors_loaded++;
    }

    // Add RoPE tables
    for (int l = 0; l < meta.n_layers; l++) {
        auto add_rope = [&](const std::string& name, float* data) {
            WeightTensor wt;
            wt.quant_type = QuantType::FP32;
            wt.n_dim = 2;
            wt.shape[0] = max_seq; wt.shape[1] = half;
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

    fclose(f);
    fprintf(stderr, "  loaded: %d tensors (cfg: %dL %ddim %dH %dKV %dhd %dV %dseq, buffer=%zuMB)\n",
            n_tensors_loaded, cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim, cfg.vocab_size, cfg.max_seq_len, total_alloc / 1048576);
    if (cfg.dim == 0 || cfg.n_layers == 0) { fprintf(stderr,"ERROR: incomplete config\n"); return false; }
    if (n_tensors_loaded == 0) { fprintf(stderr,"ERROR: no tensors loaded\n"); return false; }
    return true;
}
