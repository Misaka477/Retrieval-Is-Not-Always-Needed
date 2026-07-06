#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

enum class QuantType : int {
    Q2_1 = 1, Q1_0 = 2, Q2K_Q1V_LSC_Q4 = 3, LSC_Q4 = 4, Q4_0 = 5, FP32 = 6, Q4_0F = 7,
    // GGML/GGUF types (keep on GPU as quantized blocks)
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14,
    GGML_IQ3_XXS = 18, GGML_IQ4_NL = 20, GGML_IQ3_S = 21, GGML_IQ4_XS = 23
};

// block structs only used by CUDA kernels (quantize.cu)
#ifdef __CUDACC__
struct block_q2_1 { half scale;  uint8_t data[8]; };
struct block_q1_0 { half scale;  uint32_t bits; };
struct block_q4_0 { half scale;  uint8_t data[16]; };
struct block_q4_0_f { float scale;  uint8_t data[16]; };
#endif

// Lightweight weight reference: data pointer + quant type (no tensor metadata)
struct WeightRef {
    const void* data = nullptr;
    QuantType qt = QuantType::FP32;
    const float* f32() const { return (const float*)data; }
    explicit operator bool() const { return data != nullptr; }
};

inline int elements_per_block(QuantType qt) {
    switch (qt) {
        case QuantType::Q2_1: return 32;
        case QuantType::Q1_0: return 32;
        case QuantType::Q4_0: return 32;
        case QuantType::Q4_0F: return 32;
        default: return 1;
    }
}
inline size_t q4_0_block_size() { return 18; }   // half(2) + data[16]
inline size_t q4_0f_block_size() { return 20; }  // float(4) + data[16]
inline size_t q2_1_block_size() { return 10; }   // half(2) + data[8]
inline size_t q1_0_block_size() { return 6; }    // half(2) + uint32_t(4)

inline int ggml_block_size(QuantType qt) {
    switch (qt) {
        case QuantType::GGML_Q2_K: return 256;
        case QuantType::GGML_Q3_K: return 256;
        case QuantType::GGML_Q4_K: return 256;
        case QuantType::GGML_Q5_K: return 256;
        case QuantType::GGML_Q6_K: return 256;
        case QuantType::GGML_IQ3_XXS: return 256;
        case QuantType::GGML_IQ3_S: return 256;
        case QuantType::GGML_IQ4_NL: return 32;
        case QuantType::GGML_IQ4_XS: return 256;
        default: return elements_per_block(qt);
    }
}
inline int ggml_type_size(QuantType qt) {
    switch (qt) {
        case QuantType::GGML_Q2_K: return 84;
        case QuantType::GGML_Q3_K: return 110;
        case QuantType::GGML_Q4_K: return 144;
        case QuantType::GGML_Q5_K: return 176;
        case QuantType::GGML_Q6_K: return 210;
        case QuantType::GGML_IQ3_XXS: return 98;
        case QuantType::GGML_IQ3_S: return 110;
        case QuantType::GGML_IQ4_NL: return 18;
        case QuantType::GGML_IQ4_XS: return 136;
        default: return 4;
    }
}

inline size_t quantized_size(int n_elems, QuantType qt) {
    int epb = ggml_block_size(qt);
    int nb = (n_elems + epb - 1) / epb;
    int ts = ggml_type_size(qt);
    switch (qt) {
        case QuantType::Q4_0: return nb * q4_0_block_size();
        case QuantType::Q4_0F: return nb * q4_0f_block_size();
        case QuantType::Q2_1: return nb * q2_1_block_size();
        case QuantType::Q1_0: return nb * q1_0_block_size();
        default: return (size_t)nb * ts; // GGML types
    }
}
