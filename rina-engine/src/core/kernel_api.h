#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/quant.h"

// ——— Kernel API declarations ———
// 所有 kernel 的 launch 函数在这里声明

// dequant_matmul.cu / linear_q4.cu
cudaError_t dequant_matmul_q4_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);
cudaError_t dequant_matmul_q1_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);
cudaError_t dequant_matmul_q2_1(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream = 0);

// fp32 dispatch: choose kernel based on quant_type
void launch_linear_dispatch(
    const void* weight_data, QuantType quant_type,
    const float* input, float* output,
    int M, int N, int K, cudaStream_t stream = 0);

// rope.cu
void launch_rope(
    half* q, const half* cos, const half* sin,
    int B, int H, int T, int d, cudaStream_t stream = 0);
void launch_rope_flat(
    half* x, const half* cos, const half* sin,
    int B, int T, int H, int d, cudaStream_t stream = 0);

// rope_fp32.cu
void launch_rope_fp32(float*, const float*, const float*,
    int, int, int, int, cudaStream_t, int start_pos = 0);

// rms_norm.cu
void launch_rms_norm(
    half* x, const half* w,
    int B, int T, int d, float eps, cudaStream_t stream = 0);

// embedding (lm_head.cu)
void launch_embedding_q4_0(
    const void* weight, const int* idx, half* output,
    int B, int T, int d, cudaStream_t stream = 0);

// KV cache helpers (linear_q4.cu)
void launch_expand_kv_cache(
    const float* cache_k, const float* cache_v,
    float* Kf, float* Vf,
    int B, int H, int Hkv, int dh, int total_T, cudaStream_t stream = 0);
void launch_pack_q_to_full(
    const float* Q_flat, float* Qf,
    int B, int H, int dh, int T, int total_T, int start_pos, cudaStream_t stream = 0);

// ─── bf16 conversion and element-wise utilities (tensor_utils.cu) ───
void launch_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream = 0);
void launch_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t stream = 0);
void launch_copy_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, int n, cudaStream_t stream = 0);
void launch_add_bf16(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream = 0);
void launch_add_inplace_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream = 0);
void launch_silu_mul_bf16(__nv_bfloat16* o, const __nv_bfloat16* g, const __nv_bfloat16* u, int n, cudaStream_t stream = 0);

// ─── bf16 linear dispatch (linear_q4.cu) ───
void launch_linear_dispatch_bf16(
    const void* weight_data, QuantType quant_type,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream = 0);

// ─── bf16 RMSNorm (rms_norm.cu) ───
void launch_rms_norm_bf16(__nv_bfloat16* x, const float* w, int n, int d, float eps, cudaStream_t stream = 0);

// ─── bf16 RoPE (rope_fp32.cu) ───
void launch_rope_bf16(__nv_bfloat16* x, const float* cos_table, const float* sin_table,
                      int B, int T, int H, int d, cudaStream_t stream = 0, int start_pos = 0);

// ─── bf16 FlashAttention (flash_fp32.cu) ───
void launch_flash_attn_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O,
    int B, int H, int T, int dq, int dh, cudaStream_t stream = 0);
void launch_flashattn_fwd_save_stats_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K,
    const __nv_bfloat16* V, __nv_bfloat16* O,
    float* m_out, float* l_out,
    int B, int H, int T, int dq, int dh, cudaStream_t stream = 0);
void launch_transpose_attn_bf16(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int H, int T, int dh, cudaStream_t stream = 0);

// ─── KV cache quantize/dequant (quantize.cu) ───
void launch_quantize_k_fp32_to_q2_1(const float* input, void* output, int n, cudaStream_t stream = 0);
void launch_quantize_v_fp32_to_q1_0(const float* input, void* output, int n, cudaStream_t stream = 0);
void launch_dequant_k_q2_1_to_fp32(const void* input, float* output, int n, cudaStream_t stream = 0);
void launch_dequant_v_q1_0_to_fp32(const void* input, float* output, int n, cudaStream_t stream = 0);
void launch_quantize_kv_to_q4_0(const float* input, void* output, int n, cudaStream_t stream = 0);
void launch_dequant_kv_q4_0_to_fp32(const void* input, float* output, int n, cudaStream_t stream = 0);
void launch_quantize_kv_to_q8_0(const float* input, void* output, int n, cudaStream_t stream = 0);
void launch_dequant_kv_q8_0_to_fp32(const void* input, float* output, int n, cudaStream_t stream = 0);

// GGML quant format GPU dequant: quantized blocks → fp32 on GPU
void launch_dequant_ggml_blocks(const void* src, float* dst, int n_elems, QuantType qt, cudaStream_t stream = 0);

// test helpers (inertia_wave.cu, sparse_gather.cu)
void test_ssm_scan(int B, int T, int H, int d_h);
void test_sparse_gather_fa(int B, int H, int T, int d, int d_v, int K);
