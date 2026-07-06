#include "ops/silu_mul.h"

static const int BLK = 256;

__global__ void silu_mul_kernel(float* o, const float* g, const float* u, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

void launch_silu_mul(float* output, const float* gate, const float* up, int n, cudaStream_t stream) {
    silu_mul_kernel<<<(n + BLK - 1) / BLK, BLK, 0, stream>>>(output, gate, up, n);
}
