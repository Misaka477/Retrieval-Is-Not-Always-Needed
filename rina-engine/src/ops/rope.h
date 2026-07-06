#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ─── half-precision RoPE (B,H,T,d layout) ───
void launch_rope(
    half* q, const half* cos, const half* sin,
    int B, int H, int T, int d, cudaStream_t stream = 0);

// ─── half-precision flat RoPE (B*T,H,d flat row-major) ───
void launch_rope_flat(
    half* x, const half* cos, const half* sin,
    int B, int T, int H, int d, cudaStream_t stream = 0);

// ─── half-precision flat RoPE with fp32 cos/sin ───
void launch_rope_flat_f32cos(
    half* x, const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream = 0);

// ─── fp32 flat RoPE (for Llama/GQA, pairs i,i+half, with start_pos) ───
void launch_rope_fp32(float* x, const float* cos_table, const float* sin_table,
                      int B, int T, int H, int d, cudaStream_t stream,
                      int start_pos = 0);

// ─── fp32 RoPE backward (pairs i,i+half) ───
void launch_rope_bwd_fp32(float* dx, const float* dout,
    const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream);

// ─── bf16 RoPE ───
void launch_rope_bf16(__nv_bfloat16* x, const float* cos_table, const float* sin_table,
                      int B, int T, int H, int d, cudaStream_t stream, int start_pos = 0);

// ─── HF-style fp32 RoPE (pairs 2i,2i+1, matching transformers) ───
void launch_rope_fp32_hf(float* x, const float* cos_table, const float* sin_table,
                         int B, int T, int H, int d, cudaStream_t stream);

// ─── HF-style fp32 RoPE backward ───
void launch_rope_bwd_fp32_hf(float* dx, const float* dout,
    const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream);
