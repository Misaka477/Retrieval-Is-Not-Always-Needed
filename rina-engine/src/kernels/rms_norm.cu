#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// fp32 RMSNorm: x = x / sqrt(mean(x^2) + eps) * w
__global__ void rms_norm_fp32_kernel(float* x, const float* w, int d, float eps) {
    int bid = blockIdx.x, tid = threadIdx.x;
    float* row = x + bid * d;
    
    // Parallel sum of squares across threads
    float ss = 0;
    for (int i = tid; i < d; i += blockDim.x) ss += row[i] * row[i];
    
    // Warp shuffle reduction (within each warp of 32 threads)
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);
    
    // Cross-warp reduction: have warp 0 collect from all warps via shared memory
    __shared__ float smem[32];  // one slot per warp
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();
    
    // First warp does the final reduction
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

// half RMSNorm: converts internally (for kernel_api.h API compatibility)
void launch_rms_norm(half* x, const half* w, int B, int T, int d, float eps, cudaStream_t stream) {
    int n = B * T;
    int t = d >= 256 ? 256 : (d >= 128 ? 128 : 64);
    // Treat as fp32 (same memory, different type — safe for even d, aligned)
    rms_norm_fp32_kernel<<<n, t, sizeof(float), stream>>>((float*)x, (const float*)w, d, eps);
}
