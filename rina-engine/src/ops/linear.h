#pragma once
#include "core/quant.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ─── fp32 dispatch: auto-select kernel based on quant_type ───
void launch_linear_dispatch(
    const void* weight_data, QuantType quant_type,
    const float* input, float* output,
    int M, int N, int K, cudaStream_t stream = 0);

// ─── bf16 dispatch ───
void launch_linear_dispatch_bf16(
    const void* weight_data, QuantType quant_type,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream = 0);

// ─── GGML quant format GPU dequant: quantized blocks → fp32 on GPU ───
void launch_dequant_ggml_blocks(const void* src, float* dst, int n_elems, QuantType qt, cudaStream_t stream = 0);

// ─── Q4_0 matmuls (explicit variants for tests) ───
cudaError_t dequant_matmul_q4_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);

cudaError_t dequant_matmul_q1_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);

cudaError_t dequant_matmul_q2_1(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);

// ─── KV cache helpers (shared across GQA/MLA layers) ───
void launch_expand_kv_cache(
    const float* cache_k, const float* cache_v,
    float* Kf, float* Vf,
    int B, int H, int Hkv, int dh, int total_T, cudaStream_t stream = 0);

void launch_pack_q_to_full(
    const float* Q_flat, float* Qf,
    int B, int H, int dh, int T, int total_T, int start_pos, cudaStream_t stream = 0);

// ─── Global pre-allocated tmp buffer for dequant+matmul ───
extern float* g_dequant_tmp;
extern size_t g_dequant_tmp_sz;
