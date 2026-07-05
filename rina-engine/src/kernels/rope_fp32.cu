#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

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
                       int start_pos = 0) {
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
    const float* cos, const float* sin, int B, int T, int H, int d,
    cudaStream_t stream) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_bwd_kernel<<<grid, block, 0, stream>>>(dx, dout, cos, sin, B, T, H, d);
}

// ─── bf16 RoPE ───
// x layout: [B*T*H, d] = [n*H, d] in bf16
// cos/sin tables remain fp32
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
