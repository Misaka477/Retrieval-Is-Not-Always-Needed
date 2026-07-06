#pragma once
#include "core/quant.h"
#include <cuda_runtime.h>

// Unified embedding: supports fp32 and quantized (Q4_0, Q4_0F) weights.
// For unsupported quant types, falls back internally (requires fp32 conversion).
void launch_embedding(
    const void* weight, QuantType quant_type,
    const int* idx, float* output,
    int B, int T, int d, cudaStream_t stream = 0);

// Legacy fp32 embedding (backward compat)
void launch_embedding_fp32(const float* weight, const int* idx, float* output,
                           int B, int T, int d, cudaStream_t stream = 0);

// Embedding backward (training)
void launch_embedding_bwd_fp32(const float* dout, const int* idx,
    float* d_weight, int B, int T, int d, cudaStream_t stream = 0);
