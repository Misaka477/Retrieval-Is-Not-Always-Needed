#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

enum class QuantType : int {
    Q2_1 = 1, Q1_0 = 2, Q2K_Q1V_LSC_Q4 = 3, LSC_Q4 = 4, Q4_0 = 5, FP32 = 6
};

// block structs only used by CUDA kernels (quantize.cu)
#ifdef __CUDACC__
struct block_q2_1 { half scale;  uint8_t data[8]; };
struct block_q1_0 { half scale;  uint32_t bits; };
struct block_q4_0 { half scale;  uint8_t data[16]; };
#endif

inline int elements_per_block(QuantType qt) {
    switch (qt) {
        case QuantType::Q2_1: return 32;
        case QuantType::Q1_0: return 32;
        case QuantType::Q4_0: return 32;
        default: return 1;
    }
}
inline size_t quantized_size(int n_elems, QuantType qt) {
#ifdef __CUDACC__
    return (n_elems / elements_per_block(qt)) * (qt == QuantType::Q4_0 ? sizeof(block_q4_0) :
           qt == QuantType::Q2_1 ? sizeof(block_q2_1) : sizeof(block_q1_0));
#else
    return n_elems * sizeof(float);
#endif
}
