#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

static const int BLK_TU = 256;

// fp32 → bf16 conversion
__global__ void fp32_to_bf16_k(const float* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

void launch_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    fp32_to_bf16_k<<<grid, BLK_TU, 0, stream>>>(src, dst, n);
}

// bf16 → fp32 conversion
__global__ void bf16_to_fp32_k(const __nv_bfloat16* src, float* dst, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

void launch_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    bf16_to_fp32_k<<<grid, BLK_TU, 0, stream>>>(src, dst, n);
}

// bf16 element-wise copy
__global__ void copy_bf16_k(const __nv_bfloat16* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

void launch_copy_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    copy_bf16_k<<<grid, BLK_TU, 0, stream>>>(src, dst, n);
}

// bf16 element-wise add: c = a + b
__global__ void add_bf16_k(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) c[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

void launch_add_bf16(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    add_bf16_k<<<grid, BLK_TU, 0, stream>>>(c, a, b, n);
}

// bf16 in-place add: a += b
__global__ void add_inplace_bf16_k(__nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) a[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

void launch_add_inplace_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    add_inplace_bf16_k<<<grid, BLK_TU, 0, stream>>>(a, b, n);
}

// bf16 SiLU gate multiply: o = silu(g) * u
__global__ void silu_mul_bf16_k(__nv_bfloat16* o, const __nv_bfloat16* g, const __nv_bfloat16* u, int n) {
    int i = blockIdx.x * BLK_TU + threadIdx.x;
    if (i < n) {
        float gf = __bfloat162float(g[i]);
        float uf = __bfloat162float(u[i]);
        o[i] = __float2bfloat16((gf / (1.0f + expf(-gf))) * uf);
    }
}

void launch_silu_mul_bf16(__nv_bfloat16* o, const __nv_bfloat16* g, const __nv_bfloat16* u, int n, cudaStream_t stream) {
    int grid = (n + BLK_TU - 1) / BLK_TU;
    silu_mul_bf16_k<<<grid, BLK_TU, 0, stream>>>(o, g, u, n);
}
