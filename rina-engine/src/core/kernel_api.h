#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/quant.h"
#include "ops/linear.h"
#include "ops/rms_norm.h"
#include "ops/rope.h"
#include "ops/embedding.h"
#include "ops/silu_mul.h"
#include "ops/saxpy.h"

// ——— Kernel API declarations ———
// 所有 kernel 的 launch 函数在这里声明

// embedding (lm_head.cu)
void launch_embedding_q4_0(
    const void* weight, const int* idx, half* output,
    int B, int T, int d, cudaStream_t stream = 0);

// ─── bf16 conversion and element-wise utilities (tensor_utils.cu) ───
void launch_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream = 0);
void launch_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t stream = 0);
void launch_copy_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, int n, cudaStream_t stream = 0);
void launch_add_bf16(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream = 0);
void launch_add_inplace_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream = 0);
void launch_silu_mul_bf16(__nv_bfloat16* o, const __nv_bfloat16* g, const __nv_bfloat16* u, int n, cudaStream_t stream = 0);

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

// test helpers (inertia_wave.cu, sparse_gather.cu)
void test_ssm_scan(int B, int T, int H, int d_h);
void test_sparse_gather_fa(int B, int H, int T, int d, int d_v, int K);
