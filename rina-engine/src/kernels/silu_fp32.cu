#include <cuda_runtime.h>
#include <cmath>

// fp32 SiLU: y[i] = x[i] / (1 + exp(-x[i]))
__global__ void silu_fp32_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void launch_silu_fp32(float* x, int n, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    silu_fp32_kernel<<<grid, block, 0, stream>>>(x, n);
}
