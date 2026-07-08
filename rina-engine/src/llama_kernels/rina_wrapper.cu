// Wrapper: include llama.cpp MMVQ kernel directly, expose C-compatible entry point
// This gives us the EXACT same computation as llama.cpp for ALL quant types

#include <cuda_runtime.h>
#include <cstdio>

// GGML stub must be included before any GGML headers
#include "ggml_stub.h"

// Include the actual llama.cpp MMVQ kernel (static __global__, needs GGML stubs)
// This provides mul_mat_vec_q kernel for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, and Q8_1 input quantization
#include "mmvq.cu"

// Also include Q8_1 quantization
#include "quantize.cu"

// Simple entry point: quantize input → MMVQ → return result
// Returns a single output value for the given (weight, input, output_idx)
float compute_dot_product(const void* weight, const float* input, int K,
                          int output_idx, int quant_type, cudaStream_t stream) {
    // This function would need to:
    // 1. Quantize input to Q8_1
    // 2. Call vec_dot_qX_K_q8_1
    // 3. Return result
    
    // Actually it's simpler to just use the full MMVQ kernel
    return 0.0f;
}
