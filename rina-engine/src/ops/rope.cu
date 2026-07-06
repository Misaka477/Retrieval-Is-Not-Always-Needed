#include "ops/rope.h"
#include <cmath>

// ════════════════════════════════════════════════════════════════
// Half-precision RoPE (from kernels/rope.cu)
// ════════════════════════════════════════════════════════════════

// q: [B, H, T, d]  in-place rotary
// cos/sin: [T, d/2]
__global__ void rope_kernel(
    half* __restrict__ q,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    int B, int H, int T, int d
) {
    int half_d = d / 2;
    int total_pairs = B * H * T * half_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int pair = idx % half_d;
    int tmp  = idx / half_d;
    int pos  = tmp % T; tmp /= T;
    int head = tmp % H; tmp /= H;
    int batch = tmp;

    int base = ((batch * H + head) * T + pos) * d;
    float x0 = __half2float(q[base + pair * 2]);
    float x1 = __half2float(q[base + pair * 2 + 1]);
    float c  = __half2float(cos[pos * half_d + pair]);
    float s  = __half2float(sin[pos * half_d + pair]);

    q[base + pair * 2]     = __float2half(x0 * c - x1 * s);
    q[base + pair * 2 + 1] = __float2half(x0 * s + x1 * c);
}

void launch_rope(
    half* q, const half* cos, const half* sin,
    int B, int H, int T, int d, cudaStream_t stream
) {
    int half_d = d / 2;
    int total = B * H * T * half_d;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_kernel<<<grid, block, 0, stream>>>(q, cos, sin, B, H, T, d);
}

// Flat RoPE: x in [B*T, H*d] flat row-major layout
__global__ void rope_flat_kernel(
    half* __restrict__ x,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    int B, int T, int H, int d
) {
    int half_d = d / 2;
    int total = B * T * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int h = idx % H;
    int t = (idx / H) % T;
    int b = idx / (T * H);

    int base = ((b * T + t) * H + h) * d;
    for (int i = 0; i < half_d; i++) {
        float x0 = __half2float(x[base + i * 2]);
        float x1 = __half2float(x[base + i * 2 + 1]);
        float c  = __half2float(cos[t * half_d + i]);
        float s  = __half2float(sin[t * half_d + i]);
        x[base + i * 2]     = __float2half(x0 * c - x1 * s);
        x[base + i * 2 + 1] = __float2half(x0 * s + x1 * c);
    }
}

void launch_rope_flat(
    half* x, const half* cos, const half* sin,
    int B, int T, int H, int d, cudaStream_t stream
) {
    int total = B * T * H;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_flat_kernel<<<grid, block, 0, stream>>>(x, cos, sin, B, T, H, d);
}

// RoPE with fp32 cos/sin tables
__global__ void rope_flat_f32cos_kernel(
    half* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int B, int T, int H, int d
) {
    int half_d = d / 2;
    int total = B * T * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % H, t = (idx / H) % T, b = idx / (T * H);
    int base = ((b * T + t) * H + h) * d;
    for (int i = 0; i < half_d; i++) {
        float x0 = __half2float(x[base + i * 2]);
        float x1 = __half2float(x[base + i * 2 + 1]);
        float c = cos[t * half_d + i];
        float s = sin[t * half_d + i];
        x[base + i * 2]     = __float2half(x0 * c - x1 * s);
        x[base + i * 2 + 1] = __float2half(x0 * s + x1 * c);
    }
}

void launch_rope_flat_f32cos(
    half* x, const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream
) {
    int total = B * T * H;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_flat_f32cos_kernel<<<grid, block, 0, stream>>>(x, cos, sin, B, T, H, d);
}

// ════════════════════════════════════════════════════════════════
// fp32/bf16 RoPE (from kernels/rope_fp32.cu)
// ════════════════════════════════════════════════════════════════

// fp32 flat RoPE for Llama/GQA: pairs (i, i+half)
// x layout: [B*T*H, d] = [n*H, d], where n=B*T
// cos[T,d/2], sin[T,d/2]
__global__ void rope_fp32_kernel(float* x, const float* cos, const float* sin,
                                  int B, int T, int H, int d, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * H;
    if (idx >= n) return;
    int t = start_pos + ((idx / H) % T);
    float* row = x + idx * d;
    int half = d / 2;
    for (int i = threadIdx.y; i < half; i += blockDim.y) {
        float c = cos[t * half + i];
        float s = sin[t * half + i];
        float x0 = row[i], x1 = row[i + half];
        row[i]        = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

void launch_rope_fp32(float* x, const float* cos_table, const float* sin_table,
                       int B, int T, int H, int d, cudaStream_t stream,
                       int start_pos) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_fp32_kernel<<<grid, block, 0, stream>>>(x, cos_table, sin_table, B, T, H, d, start_pos);
}

// RoPE backward: inverse rotation (Llama: i, i+half)
__global__ void rope_bwd_kernel(float* dx, const float* dout,
    const float* cos, const float* sin, int B, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * H;
    if (idx >= n) return;
    int t = (idx / H) % T;
    const float* drow = dout + idx * d;
    float* dxrow = dx + idx * d;
    int half = d / 2;
    for (int i = threadIdx.y; i < half; i += blockDim.y) {
        float c = cos[t * half + i];
        float s = sin[t * half + i];
        float do0 = drow[i], do1 = drow[i + half];
        dxrow[i]        = do0 * c + do1 * s;
        dxrow[i + half] = -do0 * s + do1 * c;
    }
}

void launch_rope_bwd_fp32(float* dx, const float* dout,
    const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_bwd_kernel<<<grid, block, 0, stream>>>(dx, dout, cos, sin, B, T, H, d);
}

// ─── bf16 RoPE ───
__global__ void rope_bf16_kernel(__nv_bfloat16* x, const float* cos, const float* sin,
                                 int B, int T, int H, int d, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * H;
    if (idx >= n) return;
    int t = start_pos + ((idx / H) % T);
    __nv_bfloat16* row = x + idx * d;
    int half = d / 2;
    for (int i = threadIdx.y; i < half; i += blockDim.y) {
        float c = cos[t * half + i];
        float s = sin[t * half + i];
        float x0 = __bfloat162float(row[i]);
        float x1 = __bfloat162float(row[i + half]);
        row[i]        = __float2bfloat16(x0 * c - x1 * s);
        row[i + half] = __float2bfloat16(x0 * s + x1 * c);
    }
}

void launch_rope_bf16(__nv_bfloat16* x, const float* cos_table, const float* sin_table,
                      int B, int T, int H, int d, cudaStream_t stream, int start_pos) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_bf16_kernel<<<grid, block, 0, stream>>>(x, cos_table, sin_table, B, T, H, d, start_pos);
}

// ════════════════════════════════════════════════════════════════
// HF-style fp32 RoPE (from kernels/rope_fp32_hf.cu)
// ════════════════════════════════════════════════════════════════

// fp32 flat RoPE for HF-style models (L3X/Jamba): pairs (2i, 2i+1)
__global__ void rope_hf_fp32_kernel(float* x, const float* cos, const float* sin,
                                     int B, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * H;
    if (idx >= n) return;
    int t = (idx / H) % T;
    float* row = x + idx * d;
    int half = d / 2;
    for (int i = threadIdx.y; i < half; i += blockDim.y) {
        float c = cos[t * half + i];
        float s = sin[t * half + i];
        float x0 = row[2 * i], x1 = row[2 * i + 1];
        row[2 * i]       = x0 * c - x1 * s;
        row[2 * i + 1]   = x0 * s + x1 * c;
    }
}

void launch_rope_fp32_hf(float* x, const float* cos_table, const float* sin_table,
                          int B, int T, int H, int d, cudaStream_t stream) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_hf_fp32_kernel<<<grid, block, 0, stream>>>(x, cos_table, sin_table, B, T, H, d);
}

// HF-style RoPE backward: inverse rotation (HF: 2i, 2i+1)
__global__ void rope_hf_bwd_fp32_kernel(float* dx, const float* dout,
    const float* cos, const float* sin, int B, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * H;
    if (idx >= n) return;
    int t = (idx / H) % T;
    const float* drow = dout + idx * d;
    float* dxrow = dx + idx * d;
    int half = d / 2;
    for (int i = threadIdx.y; i < half; i += blockDim.y) {
        float c = cos[t * half + i];
        float s = sin[t * half + i];
        float do0 = drow[2 * i], do1 = drow[2 * i + 1];
        dxrow[2 * i]       = do0 * c + do1 * s;
        dxrow[2 * i + 1]   = -do0 * s + do1 * c;
    }
}

void launch_rope_bwd_fp32_hf(float* dx, const float* dout,
    const float* cos, const float* sin,
    int B, int T, int H, int d, cudaStream_t stream) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_hf_bwd_fp32_kernel<<<grid, block, 0, stream>>>(dx, dout, cos, sin, B, T, H, d);
}
