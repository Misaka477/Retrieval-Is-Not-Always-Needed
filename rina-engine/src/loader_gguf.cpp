// loader_gguf.cpp — Direct GGUF model loader
// Reads GGUF format models (llama.cpp ecosystem), dequantizes on-the-fly,
// outputs the same TensorMap interface as loader.cpp.
#include "core/config.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "core/json.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>

using json = nlohmann::json;

// ── GGUF constants ─────────────────────────────────────────────
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"

// GGML quantization types (relevant subset)
enum GGMLType : int {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
};

// GGUF value types for metadata
enum GGUFValueType : int {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

static int ggml_block_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 1;
        case GGML_TYPE_F16:    return 1;
        case GGML_TYPE_Q4_0:   return 32;
        case GGML_TYPE_Q4_1:   return 32;
        case GGML_TYPE_Q5_0:   return 32;
        case GGML_TYPE_Q5_1:   return 32;
        case GGML_TYPE_Q8_0:   return 32;
        case GGML_TYPE_Q2_K:   return 256;
        case GGML_TYPE_Q3_K:   return 256;
        case GGML_TYPE_Q4_K:   return 256;
        case GGML_TYPE_Q5_K:   return 256;
        case GGML_TYPE_Q6_K:   return 256;
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_IQ4_XS: return 256;
        case GGML_TYPE_IQ3_S:  return 256;
        case GGML_TYPE_IQ2_S:  return 256;
        default: return 1;
    }
}

static int ggml_type_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:    return 4;
        case GGML_TYPE_F16:    return 2;
        case GGML_TYPE_Q4_0:   return 18;
        case GGML_TYPE_Q4_1:   return 20;
        case GGML_TYPE_Q5_0:   return 22;
        case GGML_TYPE_Q5_1:   return 24;
        case GGML_TYPE_Q8_0:   return 34;
        case GGML_TYPE_Q2_K:   return 84;
        case GGML_TYPE_Q3_K:   return 110;
        case GGML_TYPE_Q4_K:   return 144;
        case GGML_TYPE_Q5_K:   return 176;
        case GGML_TYPE_Q6_K:   return 210;
        case GGML_TYPE_IQ4_NL: return 18;
        case GGML_TYPE_IQ4_XS: return 136;
        case GGML_TYPE_IQ3_S:  return 110;
        case GGML_TYPE_IQ2_S:  return 82;
        default: return 4;
    }
}

// ── GGUF file parser ───────────────────────────────────────────

struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    int ggml_type;
    uint64_t offset;  // from start of file
};

static bool read_gguf_header(const std::string& path,
                              std::vector<GGUFTensorInfo>& tensors,
                              uint64_t& data_start, int& n_layers,
                              int& dim, int& n_heads, int& n_kv_heads,
                              int& head_dim, int& vocab_size, int& max_seq_len,
                              int& d_c, int& d_h_r, int& dq, int& v_head_dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    // Magic + version
    uint32_t magic; f.read((char*)&magic, 4);
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Bad magic: 0x%x\n", magic); return false; }
    uint32_t version; f.read((char*)&version, 4);
    if (version < 2 || version > 3) { fprintf(stderr, "Unsupported GGUF version: %u\n", version); return false; }

    uint64_t n_tensors, n_kv;
    f.read((char*)&n_tensors, 8);
    f.read((char*)&n_kv, 8);

    // Read metadata KV pairs
    n_layers = 0; dim = 0; n_heads = 0; n_kv_heads = 0;
    head_dim = 0; vocab_size = 0; max_seq_len = 0; d_c = 0; dq = 0;
    std::string tokenizer_json;

    fprintf(stderr, "  parsing %llu metadata KV pairs...\n", (unsigned long long)n_kv);
    for (uint64_t i = 0; i < n_kv; i++) {
        // Key string
        uint64_t key_len; f.read((char*)&key_len, 8);
        if (key_len > 10000) { fprintf(stderr,"bad key len %llu\n",(unsigned long long)key_len); return false; }
        std::string key(key_len, '\0'); f.read(&key[0], key_len);
        // Value type
        uint32_t val_type; f.read((char*)&val_type, 4);

        auto read_str = [&]() {
            uint64_t len; f.read((char*)&len, 8);
            if (len > 1000000) { f.seekg(len, std::ios::cur); return std::string(); }
            std::string s(len, '\0');
            if (len > 0) f.read(&s[0], len);
            return s;
        };
        auto skip_arr = [&](uint32_t arr_type, uint64_t arr_len) {
            int elem_size = 4;
            if (arr_type == GGUF_TYPE_STRING) elem_size = -1;  // variable
            else if (arr_type == GGUF_TYPE_UINT8 || arr_type == GGUF_TYPE_INT8) elem_size = 1;
            else if (arr_type == GGUF_TYPE_UINT16 || arr_type == GGUF_TYPE_INT16) elem_size = 2;
            else if (arr_type == GGUF_TYPE_UINT32 || arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_FLOAT32) elem_size = 4;
            else if (arr_type == GGUF_TYPE_UINT64 || arr_type == GGUF_TYPE_INT64 || arr_type == GGUF_TYPE_FLOAT64) elem_size = 8;
            else if (arr_type == GGUF_TYPE_BOOL) elem_size = 1;
            if (elem_size < 0) {
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t sl; f.read((char*)&sl, 8);
                    f.seekg(sl, std::ios::cur);
                }
            } else {
                f.seekg((std::streamoff)((uint64_t)elem_size * arr_len), std::ios::cur);
            }
        };

        // Track architecture prefix (e.g., "llama.", "rina-jamba.")
        static std::string arch_prefix = "";
        if (key == "general.architecture") {
            arch_prefix = read_str() + ".";
        } else if (val_type == GGUF_TYPE_STRING) {
            auto val = read_str();
            if (key == "tokenizer.ggml.model") tokenizer_json = val;
        } else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
            uint32_t val; f.read((char*)&val, 4);
            auto dot = key.rfind('.');
            auto suff = (dot != std::string::npos) ? key.substr(dot + 1) : key;
            if (suff == "block_count") { n_layers = val; fprintf(stderr,"  n_layers=%d\n",n_layers); }
            if (suff == "embedding_length") { dim = val; fprintf(stderr,"  dim=%d\n",dim); }
            if (suff == "head_count") { n_heads = val; }
            if (suff == "head_count_kv") { n_kv_heads = val; }
            if (suff == "dimension_count") { d_h_r = val; head_dim = val; fprintf(stderr,"  rope.dim=%d\n",d_h_r); }
            if (suff == "context_length") { max_seq_len = val; }
            if (suff == "embedding_length") dim = val;
            if (suff == "head_count") n_heads = val;
            if (suff == "head_count_kv") n_kv_heads = val;
            if (suff == "dimension_count") { d_h_r = val; head_dim = val; }
            if (suff == "context_length") max_seq_len = val;
            if (suff == "vocab_size") vocab_size = val;
            if (suff == "kv_lora_rank") { d_c = val; fprintf(stderr,"  kv_lora_rank=%d\n",d_c); }
            if (suff == "key_length") { dq = val; fprintf(stderr,"  key_length=%d\n",dq); }
            if (suff == "value_length") { v_head_dim = val; fprintf(stderr,"  value_length=%d\n",v_head_dim); }
        } else if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type; f.read((char*)&arr_type, 4);
            uint64_t arr_len; f.read((char*)&arr_len, 8);
            skip_arr(arr_type, arr_len);
        } else {
            // Skip other types
            int skip = 0;
            if (val_type == GGUF_TYPE_UINT64 || val_type == GGUF_TYPE_INT64 || val_type == GGUF_TYPE_FLOAT64) skip = 8;
            else if (val_type == GGUF_TYPE_FLOAT32) skip = 4;
            else if (val_type == GGUF_TYPE_BOOL) skip = 1;
            f.seekg(skip, std::ios::cur);
        }
    }

    // Compute data section start (file position after all tensor infos + padding)
    // First, skip the tensor info entries to find the data section offset
    uint64_t tensor_info_start = (uint64_t)f.tellg();
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len; f.read((char*)&name_len, 8);
        f.seekg(name_len, std::ios::cur);
        uint32_t n_dim; f.read((char*)&n_dim, 4);
        f.seekg((std::streamoff)n_dim * 8, std::ios::cur);
        f.seekg(4 + 8, std::ios::cur); // type + offset (no size field)
    }
    uint64_t alignment = 32;
    data_start = (uint64_t)f.tellg();
    data_start = (data_start + alignment - 1) / alignment * alignment;

    // Now re-read tensor infos with the correct data_start
    f.seekg(tensor_info_start, std::ios::beg);
    tensors.clear();
    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo ti;
        uint64_t name_len; f.read((char*)&name_len, 8);
        ti.name.resize(name_len); f.read(&ti.name[0], name_len);
        uint32_t n_dim; f.read((char*)&n_dim, 4);
        ti.shape.resize(n_dim);
        for (int j = 0; j < (int)n_dim; j++) {
            uint64_t d; f.read((char*)&d, 8);
            ti.shape[j] = (int64_t)d;
        }
        uint32_t gtype; f.read((char*)&gtype, 4);
        ti.ggml_type = gtype;
        f.read((char*)&ti.offset, 8);  // GGUF offset is relative to data_start
        ti.offset += data_start;  // convert to absolute file offset
        tensors.push_back(ti);
    }
    return true;
}

// ── HF→RINN name mapping (reuses logic from loader_hf) ─────────

static bool is_layer_weight(const std::string& name) {
    return name.rfind("blk.", 0) == 0;
}

static int layer_index(const std::string& gguf_name) {
    // "blk.0.attn_q.weight" → 0,  "output_norm.weight" → -1
    if (gguf_name.rfind("blk.", 0) != 0) return -1;
    auto p = gguf_name.find('.', 4);
    return std::stoi(gguf_name.substr(4, p - 4));
}

static std::string gguf_to_rinn_name(const std::string& gguf_name, int l) {
    // GGUF names → RINN names
    // Attention: blk.{l}.attn_{q,k,v,o}.weight → transformer.h.{l}.attn.w_{q,k,v,o}.weight
    if (gguf_name == "token_embd.weight")    return "transformer.wte.weight";
    if (gguf_name == "output_norm.weight")   return "transformer.ln_f.weight";
    if (gguf_name == "output.weight")        return "lm_head.weight";

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

// Shared non-QAT list from loader_hf
static bool should_quant(const std::string& rinn_name) {
    if (rinn_name.find("wte") != std::string::npos) return false;
    if (rinn_name.find("ln") != std::string::npos) return false;
    if (rinn_name.find("lm_head") != std::string::npos) return false;
    if (rinn_name.find("rope") != std::string::npos) return false;
    if (rinn_name.find("norm") != std::string::npos) return false;
    return rinn_name.find(".weight") != std::string::npos;
}

// ── Dequantization helpers ─────────────────────────────────────

constexpr int QK_K = 256;

static inline float half_to_float(uint16_t h) {
    // Standard IEEE-754 half-precision to single-precision conversion
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        // Denormalized: value = (-1)^sign * 2^(1-15) * mant/1024
        // Convert to normalized fp32 by finding first 1 in mantissa
        uint32_t m = mant;
        uint32_t e = 113;  // 127 - 14 = 113 (exponent for 2^(-14))
        if (m == 0) {
            f32 = (sign << 31);  // zero
        } else {
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            f32 = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0xFF << 23 | (mant << 13);
        if (mant) f32 |= 0x7FFFFF;  // NaN
    } else {
        f32 = (sign << 31) | (exp + 127 - 15) << 23 | mant << 13;
    }
    float r; memcpy(&r, &f32, 4); return r;
}

// IQ4_NL k-values (for IQ4_XS sub-blocks)
static const float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f
};

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequant_q4_K(const uint8_t* src, float* dst, int n) {
    // Q4_K: 2 halves + 12 scale bytes + 128 data = 144 bytes per 256 values
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 144;
        float d = half_to_float(*(const uint16_t*)(blk));
        float min = half_to_float(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        int is = 0;
        for (int j = 0; j < BLK && b * BLK + j < n; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            int off = b * BLK + j;
            for (int l = 0; l < 32 && off + l < n; l++)
                dst[off + l]      = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32 && off + 32 + l < n; l++)
                dst[off + 32 + l] = d2 * (qs[l] >> 4) - m2;
            qs += 32; is += 2;
        }
    }
}

static void dequant_q5_K(const uint8_t* src, float* dst, int n) {
    // Q5_K: 2 halves + 12 scale + 128 low bits + 32 high bits = 176 bytes per 256 values
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 176;
        float d = half_to_float(*(const uint16_t*)(blk));
        float min = half_to_float(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        int is = 0;
        for (int j = 0; j < BLK && b * BLK + j < n; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            int off = b * BLK + j;
            for (int l = 0; l < 32 && off + l < n; l++) {
                int ql_l = ql[l] & 0xF;
                int qh_bit = (qh[l/2] >> ((l % 2) * 4)) & 1;
                dst[off + l] = d1 * ((ql_l | (qh_bit << 4))) - m1;
            }
            for (int l = 0; l < 32 && off + 32 + l < n; l++) {
                int ql_l = ql[l] >> 4;
                int qh_bit = (qh[l/2] >> ((l % 2) * 4 + 4)) & 1;  // wrong offset
                dst[off + 32 + l] = d2 * ((ql_l | (qh_bit << 4))) - m2;
            }
            ql += 32; qh += 16; is += 2;
        }
    }
}

static void dequant_q6_K(const uint8_t* src, float* dst, int n) {
    // Q6_K: 128 ql + 64 qh + 16 scales + 2 half = 210 bytes per 256 values
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 210;
        float d = half_to_float(*(const uint16_t*)(blk + 208));  // d is at bytes 208-209
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);  // scales are at bytes 192-207
        for (int nblk = 0; nblk < BLK && b * BLK + nblk < n; nblk += 128) {
            int off = b * BLK + nblk;
            for (int l = 0; l < 32 && off + l < n; l++) {
                int is = l / 16;
                int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[off + l]       = d * sc[is + 0] * q1;
                dst[off + l + 32]  = d * sc[is + 2] * q2;
                dst[off + l + 64]  = d * sc[is + 4] * q3;
                dst[off + l + 96]  = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32;
        }
    }
}

static void dequant_iq4_xs(const uint8_t* src, float* dst, int n) {
    // IQ4_XS: 2 half + 2 scales_h + 4 scales_l + 128 qs = 136 bytes per 256 values
    const int BLK = 256;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 136;
        float d = half_to_float(*(const uint16_t*)(blk));
        uint16_t scales_h = *(const uint16_t*)(blk + 2);
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        int off = b * BLK;
        for (int ib = 0; ib < BLK/32 && off + ib * 32 < n; ib++) {
            int ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf;
            ls |= ((scales_h >> 2*ib) & 3) << 4;
            float dl = d * (ls - 32);
            for (int j = 0; j < 16 && off + ib * 32 + j < n; j++)
                dst[off + ib * 32 + j]      = dl * kvalues_iq4nl[qs[j] & 0xf];
            for (int j = 0; j < 16 && off + ib * 32 + 16 + j < n; j++)
                dst[off + ib * 32 + 16 + j] = dl * kvalues_iq4nl[qs[j] >> 4];
            qs += 16;
        }
    }
}

static void dequant_block_f16(const uint8_t* src, float* dst, int n) {
    const uint16_t* h = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        uint32_t sign = (h[i] >> 15) & 1;
        uint32_t exp = (h[i] >> 10) & 0x1F;
        uint32_t mant = h[i] & 0x3FF;
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

static void dequant_q4_0(const uint8_t* src, float* dst, int n) {
    // Q4_0: half scale + 16 bytes data = 18 bytes for 32 values
    const int BLK = 32;
    int nb = (n + BLK - 1) / BLK;
    for (int b = 0; b < nb; b++) {
        uint16_t scale_half; memcpy(&scale_half, src + b * 18, 2);
        // Half to float
        int sign = (scale_half >> 15) & 1;
        int exp = (scale_half >> 10) & 0x1F;
        int mant = scale_half & 0x3FF;
        float scale;
        uint32_t f32 = (sign << 31) | ((exp == 0 ? 0 : exp + 127 - 15) << 23) | (mant << 13);
        if (exp == 0) f32 = (sign << 31) | (0x7F - 15 + 1) << 23 | mant << 13;
        memcpy(&scale, &f32, 4);

        const uint8_t* data = src + b * 18 + 2;
        for (int i = 0; i < 32 && b * 32 + i < n; i++) {
            int q = (data[i / 2] >> ((i & 1) << 2)) & 0xF;
            dst[b * 32 + i] = (q - 7) * scale;
        }
    }
}

// ── Main GGUF loader ──────────────────────────────────────────

bool load_gguf_model(const char* path, ModelConfig& cfg, TensorMap& tensors, int max_layers) {
    std::vector<GGUFTensorInfo> gguf_tensors;
    uint64_t data_start;
    int n_layers = 0, dim = 0, n_heads = 0, n_kv_heads = 0;
    int head_dim = 0, vocab_size = 0, max_seq_len = 0;
    int d_c = 0, d_h_r = 0, dq = 0, v_head_dim = 0;

    if (!read_gguf_header(path, gguf_tensors, data_start,
                          n_layers, dim, n_heads, n_kv_heads,
                          head_dim, vocab_size, max_seq_len, d_c, d_h_r, dq, v_head_dim))
        return false;

    bool is_deepseek = (d_c > 0 && dq > 0);

    // Build RINN config
    if (head_dim == 0) head_dim = dim / n_heads;
    if (n_kv_heads == 0) n_kv_heads = n_heads;
    if (max_seq_len == 0) max_seq_len = 2048;

    cfg.name = "gguf-llama-" + std::to_string(dim / 1000) + "dim";
    cfg.dim = dim;
    cfg.n_layers = n_layers;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.head_dim = head_dim;
    cfg.d_c = 0;
    cfg.d_h_r = head_dim;
    cfg.vocab_size = vocab_size;
    cfg.max_seq_len = std::min(max_seq_len, 1024);
    cfg.ssm_steps = 0;
    cfg.weight_tying = true;
    cfg.layers.clear();
    for (int i = 0; i < n_layers; i++)
        cfg.layers.push_back({"standard_attention", "layer_" + std::to_string(i), 1, {}});

    // Open file for tensor reads
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    // Generate RoPE tables
    int max_seq = max_seq_len;
    int half = head_dim / 2;
    std::vector<float> cos_tbl(max_seq * half);
    std::vector<float> sin_tbl(max_seq * half);
    float theta = 10000.0f;
    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        for (int t = 0; t < max_seq; t++) {
            float val = (float)t * inv_freq;
            cos_tbl[t * half + i] = cosf(val);
            sin_tbl[t * half + i] = sinf(val);
        }
    }

    if (is_deepseek) {
        cfg.name = "deepseek2-" + std::to_string(dim / 1000) + "dim";
        cfg.d_c = d_c;
        cfg.layers.clear();
        for (int i = 0; i < n_layers; i++)
            cfg.layers.push_back({"deepseek_mla_moe", "layer_" + std::to_string(i), 1, {}});
    }

    // Read and dequantize tensors
    int n_fp32 = 0, n_q4 = 0;
    for (auto& gt : gguf_tensors) {
        std::string rinn_name = gguf_to_rinn_name(gt.name, 0);
        if (rinn_name.empty()) {
            // Try lm_head / output.weight handling
            if (gt.name == "output.weight") rinn_name = "transformer.wte.weight";
            else continue;
        }
        if (max_layers > 0 && is_layer_weight(gt.name)) {
            int li = layer_index(gt.name);
            if (li >= max_layers) continue;
        }

        int64_t n_elems = 1;
        for (auto s : gt.shape) n_elems *= s;

        // Read block
        int blk_sz = ggml_block_size(gt.ggml_type);
        int type_sz = ggml_type_size(gt.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t raw_size = (size_t)n_blocks * type_sz;

        // For GGML quant types: upload raw quantized blocks to GPU directly
        bool is_embed = (gt.name == "token_embd.weight");
        if (!is_embed && (gt.ggml_type == GGML_TYPE_Q4_K ||
            gt.ggml_type == GGML_TYPE_Q6_K ||
            gt.ggml_type == GGML_TYPE_IQ4_XS ||
            gt.ggml_type == GGML_TYPE_Q5_K)) {

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
            // Map GGML type to our QuantType
            switch (gt.ggml_type) {
                case GGML_TYPE_Q4_K:  wt.quant_type = QuantType::GGML_Q4_K; break;
                case GGML_TYPE_Q5_K:  wt.quant_type = QuantType::GGML_Q5_K; break;
                case GGML_TYPE_Q6_K:  wt.quant_type = QuantType::GGML_Q6_K; break;
                case GGML_TYPE_IQ4_XS: wt.quant_type = QuantType::GGML_IQ4_XS; break;
            }
            cudaMalloc(&wt.data, raw_size);
            cudaMemcpy(wt.data, raw.data(), raw_size, cudaMemcpyHostToDevice);
            tensors.add(rinn_name, std::move(wt));
            n_fp32++;
            continue;
        }

        // F32/F16/Q4_0: CPU dequant (these are small tensors like LN, RoPE, or Q4_0)
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
        cudaMalloc(&wt.data, bytes);
        cudaMemcpy(wt.data, f32_buf.data(), bytes, cudaMemcpyHostToDevice);
        tensors.add(rinn_name, std::move(wt));
        n_fp32++;
        // debug removed
    }

    // Add RoPE tables
    for (int l = 0; l < n_layers; l++) {
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
    fprintf(stderr, "  loaded: %d tensors (cfg: %dL %ddim %dH %dKV %dhd %dV %dseq)\n",
            n_fp32, cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim, cfg.vocab_size, cfg.max_seq_len);
    if (cfg.dim == 0 || cfg.n_layers == 0) { fprintf(stderr,"ERROR: incomplete config\n"); return false; }
    if (n_fp32 == 0) { fprintf(stderr,"ERROR: no tensors loaded\n"); return false; }
    // debug removed
    return true;
}
