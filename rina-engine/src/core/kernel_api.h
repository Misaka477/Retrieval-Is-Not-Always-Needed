#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/quant.h"

// ——— Kernel API declarations ———
// 所有 kernel 的 launch 函数在这里声明

// dequant_matmul.cu
cudaError_t dequant_matmul_q4_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);
cudaError_t dequant_matmul_q1_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);
cudaError_t dequant_matmul_q2_1(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);

// rope.cu
void launch_rope(
    half* q, const half* cos, const half* sin,
    int B, int H, int T, int d, cudaStream_t stream = 0);
void launch_rope_flat(
    half* x, const half* cos, const half* sin,
    int B, int T, int H, int d, cudaStream_t stream = 0);

// rms_norm.cu
void launch_rms_norm(
    half* x, const half* w,
    int B, int T, int d, float eps, cudaStream_t stream = 0);

// embedding (lm_head.cu)
void launch_embedding_q4_0(
    const void* weight, const int* idx, half* output,
    int B, int T, int d, cudaStream_t stream = 0);

// test helpers (inertia_wave.cu, sparse_gather.cu)
void test_ssm_scan(int B, int T, int H, int d_h);
void test_sparse_gather_fa(int B, int H, int T, int d, int d_v, int K);
