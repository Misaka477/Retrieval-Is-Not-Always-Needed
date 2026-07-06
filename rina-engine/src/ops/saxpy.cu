#include "ops/saxpy.h"

static const int BLK = 256;

__global__ void saxpy_kernel(float* d, const float* s, float p, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) d[i] += p * s[i];
}

__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch_saxpy(float* dst, const float* src, float scale, int n, cudaStream_t stream) {
    saxpy_kernel<<<(n + BLK - 1) / BLK, BLK, 0, stream>>>(dst, src, scale, n);
}

void launch_add(float* c, const float* a, const float* b, int n, cudaStream_t stream) {
    add_kernel<<<(n + BLK - 1) / BLK, BLK, 0, stream>>>(c, a, b, n);
}
