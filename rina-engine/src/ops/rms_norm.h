#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

void launch_rms_norm_fp32(float* x, const float* w, int n, int d, float eps, cudaStream_t stream = 0);

void launch_rms_norm(half* x, const half* w, int B, int T, int d, float eps, cudaStream_t stream = 0);

void launch_rms_norm_bf16(__nv_bfloat16* x, const float* w, int n, int d, float eps, cudaStream_t stream = 0);
