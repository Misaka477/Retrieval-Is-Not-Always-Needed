// GGML stub header for compiling llama.cpp CUDA kernels standalone
// Provides minimal GGML types needed by the kernel files.
#ifndef GGML_STUB_H
#define GGML_STUB_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// ---- types: ggml_type enum ----
enum ggml_type {
    GGML_TYPE_F32=0, GGML_TYPE_F16=1, GGML_TYPE_Q4_0=2, GGML_TYPE_Q4_1=3,
    GGML_TYPE_Q5_0=6, GGML_TYPE_Q5_1=7, GGML_TYPE_Q8_0=8,
    GGML_TYPE_Q8_1=9, GGML_TYPE_Q2_K=10, GGML_TYPE_Q3_K=11, GGML_TYPE_Q4_K=12,
    GGML_TYPE_Q5_K=13, GGML_TYPE_Q6_K=14, GGML_TYPE_Q8_K=15,
    GGML_TYPE_IQ2_XXS=16, GGML_TYPE_IQ2_XS=17, GGML_TYPE_IQ3_XXS=18,
    GGML_TYPE_IQ3_S=19, GGML_TYPE_IQ4_NL=23, GGML_TYPE_IQ4_XS=24,
    GGML_TYPE_F16=1,
};

// ---- assertions ----
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr,"GGML_ASSERT:%s\n",#x); assert(x); } } while(0)
#define GGML_UNUSED(x) (void)(x)
#define GGML_COMMON_DECL_C
#define GGML_CUDA_PDL_RUNTIME 0

// ---- tensor ----
struct ggml_tensor {
    ggml_type type;
    void * data;
    int64_t ne[4];
    size_t nb[4];
};

// ---- kernel launch params ----
struct ggml_cuda_kernel_launch_params {
    dim3 grid; dim3 block; int smem; cudaStream_t stream;
    ggml_cuda_kernel_launch_params(dim3 g, dim3 b, int s, cudaStream_t st):grid(g),block(b),smem(s),stream(st){}
};

template<typename F, typename... Args>
void ggml_cuda_kernel_launch(F kernel, const ggml_cuda_kernel_launch_params & p, Args... args) {
    kernel<<<p.grid, p.block, p.smem, p.stream>>>(args...);
}

// ---- fast mod ----
struct fastdiv_t { uint64_t M; uint8_t S1, S2; };
struct uint3 { unsigned int x, y, z; };
static inline uint3 init_fastdiv_values(int64_t v) { return {(unsigned int)v,0,0}; }
static inline unsigned int fastmodulo(int v, uint3 p) { return v % p.x; }

// ---- warp ops ----
#define WARP_SIZE 32
template<int SZ>
static __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for(int m=SZ/2;m>0;m>>=1) v+=__shfl_xor_sync(0xFFFFFFFF,v,m);
    return v;
}
template<int SZ>
static __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for(int m=SZ/2;m>0;m>>=1) v=fmaxf(v,__shfl_xor_sync(0xFFFFFFFF,v,m));
    return v;
}

// ---- block reduction (minimal) ----
enum class block_reduce_method { SUM, MAX };
template<block_reduce_method M, int BS>
__device__ float block_reduce(float v, float* shared) {
    __shared__ float tmp[32];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float r = (M==block_reduce_method::MAX)?warp_reduce_max<32>(v):warp_reduce_sum<32>(v);
    if(lane==0) tmp[warp]=r;
    __syncthreads();
    if(warp==0) r = (threadIdx.x<(blockDim.x+31)/32)?tmp[threadIdx.x]:((M==block_reduce_method::MAX)?-INFINITY:0.0f);
    if(blockDim.x>32) r=(M==block_reduce_method::MAX)?warp_reduce_max<32>(r):warp_reduce_sum<32>(r);
    return (threadIdx.x==0)?r:((M==block_reduce_method::MAX)?-INFINITY:0.0f);
}

// ---- CPU PDL ----
#define ggml_cuda_pdl_lc(...)
#define ggml_cuda_pdl_sync(...)

// ---- backend ctx (minimal) ----
struct ggml_backend_cuda_context {
    cudaStream_t stream() { return 0; }
};

// ---- CUDA block size ----
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256

// ---- misc ----
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_NAME 64

#endif
