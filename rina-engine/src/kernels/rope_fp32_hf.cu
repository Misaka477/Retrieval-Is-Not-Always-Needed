#include <cuda_runtime.h>
#include <cmath>

// fp32 flat RoPE for our models (L3X/Jamba): pairs (2i, 2i+1) matching HF/transformers
// x layout: [B*T*H, d] = [n*H, d], where n=B*T
// cos[T,d/2], sin[T,d/2]
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

// RoPE backward: inverse rotation (HF: 2i, 2i+1)
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
    const float* cos, const float* sin, int B, int T, int H, int d,
    cudaStream_t stream) {
    int n = B * T * H;
    dim3 block(128, 4);
    dim3 grid((n + block.x - 1) / block.x);
    rope_hf_bwd_fp32_kernel<<<grid, block, 0, stream>>>(dx, dout, cos, sin, B, T, H, d);
}
