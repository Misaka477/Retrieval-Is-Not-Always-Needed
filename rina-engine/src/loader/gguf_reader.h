#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

// ── GGUF constants ─────────────────────────────────────────────
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"

// GGML quantization types (relevant subset)
// Note: this enum values overlap with ggml.h's ggml_type.
// Avoid including ggml.h when this header is used, or include this header first.
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
    GGML_TYPE_IQ3_XXS = 18,
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

inline int ggml_block_size(int type) {
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
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ3_S: return 256;
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_IQ4_XS: return 256;
        case GGML_TYPE_IQ2_S:  return 256;
        default: return 1;
    }
}

inline int ggml_type_size(int type) {
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
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ3_S: return 110;
        case GGML_TYPE_IQ4_NL: return 18;
        case GGML_TYPE_IQ4_XS: return 136;
        case GGML_TYPE_IQ2_S:  return 82;
        default: return 4;
    }
}

// ── GGUF tensor info ──────────────────────────────────────────
struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    int ggml_type;
    uint64_t offset;  // absolute file offset
};

// ── GGUF metadata (architecture-agnostic parsed header) ───────
struct GGUFMetadata {
    int n_layers = 0;
    int dim = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    // MLA-specific fields
    int d_c = 0;
    int d_h_r = 0;
    int dq = 0;
    int v_head_dim = 0;
    std::string arch_string;  // e.g. "llama", "deepseek2"
    // RoPE config
    float rope_freq_base = 10000.0f;
    float rope_scaling_factor = 1.0f;  // YaRN scaling factor (1.0 = no scaling)
    std::vector<GGUFTensorInfo> tensors;
    uint64_t data_start;
};

// ── Read GGUF header from file, populate metadata and tensor list ──
bool read_gguf_header(const std::string& path, GGUFMetadata& meta);

// ── CPU dequantization helpers ──
void dequant_q4_K(const uint8_t* src, float* dst, int n);
void dequant_q5_K(const uint8_t* src, float* dst, int n);
void dequant_q6_K(const uint8_t* src, float* dst, int n);
void dequant_iq4_xs(const uint8_t* src, float* dst, int n);
void dequant_block_f16(const uint8_t* src, float* dst, int n);
void dequant_q4_0(const uint8_t* src, float* dst, int n);
