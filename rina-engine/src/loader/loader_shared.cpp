#include "loader/loader_base.h"
#include "core/quant.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

void generate_rope_tables(int max_seq, int head_dim,
                          std::vector<float>& cos_tbl,
                          std::vector<float>& sin_tbl,
                          float freq_base,
                          float yarn_scale) {
    int half = head_dim / 2;
    cos_tbl.resize(max_seq * half);
    sin_tbl.resize(max_seq * half);
    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
        if (yarn_scale > 1.0f) inv_freq /= yarn_scale;
        for (int t = 0; t < max_seq; t++) {
            float val = (float)t * inv_freq;
            cos_tbl[t * half + i] = cosf(val);
            sin_tbl[t * half + i] = sinf(val);
        }
    }
}

bool should_quant_weight(const std::string& rinn_name) {
    if (rinn_name.find("wte") != std::string::npos) return false;
    if (rinn_name.find("ln") != std::string::npos) return false;
    if (rinn_name.find("lm_head") != std::string::npos) return false;
    if (rinn_name.find("rope") != std::string::npos) return false;
    if (rinn_name.find("norm") != std::string::npos) return false;
    return rinn_name.find(".weight") != std::string::npos;
}

QuantType ggml_to_quant_type(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_Q4_0:   return QuantType::Q4_0;
        case GGML_TYPE_Q4_K:   return QuantType::GGML_Q4_K;
        case GGML_TYPE_Q5_K:   return QuantType::GGML_Q5_K;
        case GGML_TYPE_Q6_K:   return QuantType::GGML_Q6_K;
        case GGML_TYPE_IQ4_NL: return QuantType::GGML_IQ4_NL;
        case GGML_TYPE_IQ4_XS: return QuantType::GGML_IQ4_XS;
        case GGML_TYPE_Q2_K:   return QuantType::GGML_Q2_K;
        case GGML_TYPE_Q3_K:   return QuantType::GGML_Q3_K;
        default:               return QuantType::FP32;
    }
}

// ── Architecture loader registry ─────────────────────────────

static std::vector<ArchLoader> s_loaders;

void register_loader(const ArchLoader& loader) {
    s_loaders.push_back(loader);
}

const std::vector<ArchLoader>& get_registered_loaders() {
    return s_loaders;
}

const ArchLoader* detect_arch(const GGUFMetadata& meta) {
    for (auto& loader : s_loaders) {
        if (loader.detect && loader.detect(meta))
            return &loader;
    }
    return nullptr;
}

// ── Single buffer allocation ────────────────────────────────

void* allocate_weight_buffer(const std::vector<GGUFTensorInfo>& tensors,
                             size_t& total_bytes) {
    // First pass: sum all tensor sizes (raw quantized bytes)
    total_bytes = 0;
    for (auto& t : tensors) {
        int64_t n_elems = 1;
        for (auto s : t.shape) n_elems *= s;
        int blk_sz = ggml_block_size(t.ggml_type);
        int type_sz = ggml_type_size(t.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t raw_size = (size_t)n_blocks * type_sz;
        // Align to 256 bytes
        total_bytes = (total_bytes + 255) & ~255;
        total_bytes += raw_size;
    }
    // Second pass: allocate
    void* buf;
    cudaMalloc(&buf, total_bytes);
    if (!buf) {
        fprintf(stderr, "  allocate_weight_buffer OOM: %zu MB\n",
                total_bytes / 1048576);
        total_bytes = 0;
    }
    return buf;
}

size_t tensor_buffer_offset(const std::vector<GGUFTensorInfo>& tensors, int idx) {
    size_t offset = 0;
    for (int i = 0; i < idx; i++) {
        auto& t = tensors[i];
        int64_t n_elems = 1;
        for (auto s : t.shape) n_elems *= s;
        int blk_sz = ggml_block_size(t.ggml_type);
        int type_sz = ggml_type_size(t.ggml_type);
        int n_blocks = (n_elems + blk_sz - 1) / blk_sz;
        size_t raw_size = (size_t)n_blocks * type_sz;
        offset = (offset + 255) & ~255;
        offset += raw_size;
    }
    return offset;
}
