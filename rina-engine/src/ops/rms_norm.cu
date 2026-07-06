#include "ops/rms_norm.h"
#include <cmath>

// fp32 RMSNorm: x = x / sqrt(mean(x^2) + eps) * w
__global__ void rms_norm_fp32_kernel(float* x, const float* w, int d, float eps) {
    int bid = blockIdx.x, tid = threadIdx.x;
    float* row = x + bid * d;
    
    float ss = 0;
    for (int i = tid; i < d; i += blockDim.x) ss += row[i] * row[i];
    
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);
    
    __shared__ float smem[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();
    
    if (warp_id == 0) {
        ss = (tid < (blockDim.x / 32)) ? smem[tid] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);
        if (tid == 0) smem[0] = rsqrtf(ss / (float)d + eps);
    }
    __syncthreads();
    
    float inv = smem[0];
    for (int i = tid; i < d; i += blockDim.x) row[i] = row[i] * inv * w[i];
}

void launch_rms_norm_fp32(float* x, const float* w, int n, int d, float eps, cudaStream_t stream) {
    int t = d >= 256 ? 256 : (d >= 128 ? 128 : 64);
    rms_norm_fp32_kernel<<<n, t, sizeof(float), stream>>>(x, w, d, eps);
}

void launch_rms_norm(half* x, const half* w, int B, int T, int d, float eps, cudaStream_t stream) {
    int n = B * T;
    int t = d >= 256 ? 256 : (d >= 128 ? 128 : 64);
    rms_norm_fp32_kernel<<<n, t, sizeof(float), stream>>>((float*)x, (const float*)w, d, eps);
}

// ─── bf16 RMSNorm ───
__global__ void rms_norm_bf16_kernel(__nv_bfloat16* x, const float* w, int d, float eps) {
    int bid = blockIdx.x, tid = threadIdx.x;
    __nv_bfloat16* row = x + bid * d;

    float ss = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        ss += v * v;
    }
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);

    __shared__ float smem[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();

    if (warp_id == 0) {
        ss = (tid < (blockDim.x / 32)) ? smem[tid] : 0.0f;
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);
        if (tid == 0) smem[0] = rsqrtf(ss / (float)d + eps);
    }
    __syncthreads();

    float inv = smem[0];
    for (int i = tid; i < d; i += blockDim.x)
        row[i] = __float2bfloat16(__bfloat162float(row[i]) * inv * w[i]);
}

void launch_rms_norm_bf16(__nv_bfloat16* x, const float* w, int n, int d, float eps, cudaStream_t stream) {
    int t = d >= 256 ? 256 : (d >= 128 ? 128 : 64);
    rms_norm_bf16_kernel<<<n, t, 0, stream>>>(x, w, d, eps);
}
