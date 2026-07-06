#pragma once
#include <cuda_runtime.h>

// SAXPY: d[i] += scale * s[i]
void launch_saxpy(float* dst, const float* src, float scale, int n, cudaStream_t stream = 0);

// Element-wise add: c[i] = a[i] + b[i]
void launch_add(float* c, const float* a, const float* b, int n, cudaStream_t stream = 0);
