#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

inline cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    }
    return handle;
}

inline void gemm_fp32(float* out, const float* in, const float* weight,
                       int M, int N, int K, cudaStream_t stream) {
    cublasHandle_t h = get_cublas_handle();
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha,
                weight, K, in, K, &beta, out, N);
}

// Out-of-place variant: writes result to a temp buffer (tmpptr) instead of out.
// tmpptr must be at least M*N floats and NOT overlap with `in`.
inline void gemm_fp32_outofplace(float* out, const float* in, const float* weight,
                                  int M, int N, int K, float* tmpptr, cudaStream_t stream) {
    gemm_fp32(tmpptr, in, weight, M, N, K, stream);
    cudaMemcpyAsync(out, tmpptr, M*N*sizeof(float), cudaMemcpyDeviceToDevice, stream);
}
