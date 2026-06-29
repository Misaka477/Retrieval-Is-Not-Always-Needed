#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    int B, int H, int T, int d, cudaStream_t stream = 0
) {
    int half_d = d / 2;
    int total = B * H * T * half_d;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_kernel<<<grid, block, 0, stream>>>(q, cos, sin, B, H, T, d);
}

// Flat RoPE: x in [B*T, H*d] flat row-major layout
// cos/sin: [T, d/2]
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
    int B, int T, int H, int d, cudaStream_t stream = 0
) {
    int total = B * T * H;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_flat_kernel<<<grid, block, 0, stream>>>(x, cos, sin, B, T, H, d);
}

// RoPE with fp32 cos/sin tables (matching PyTorch's exact computation)
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
    int B, int T, int H, int d, cudaStream_t stream = 0
) {
    int total = B * T * H;
    int block = 256;
    int grid  = (total + block - 1) / block;
    rope_flat_f32cos_kernel<<<grid, block, 0, stream>>>(x, cos, sin, B, T, H, d);
}
