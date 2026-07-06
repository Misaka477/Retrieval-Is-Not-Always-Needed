#pragma once
#include <cuda_runtime.h>

// SiLU(x) * y element-wise: o[i] = (x[i] / (1 + exp(-x[i]))) * y[i]
void launch_silu_mul(float* output, const float* gate, const float* up, int n, cudaStream_t stream = 0);
