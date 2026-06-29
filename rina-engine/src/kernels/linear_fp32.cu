#include <cuda_runtime.h>
#include "gemm.cuh"

// Simple fp32 linear: out[M,N] = in[M,K] @ weight[N,K]^T
__global__ void linear_fp32_kernel(const float* in, const float* weight, float* out, int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += in[m * K + k] * weight[n * K + k];
    out[m * N + n] = sum;
}

void launch_linear_fp32(const float* in, const float* weight, float* out, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M+15)/16, (N+15)/16);
    if (in == out) {
        float* tmp;
        cudaMalloc(&tmp, M*N*sizeof(float));
        linear_fp32_kernel<<<grid, block, 0, stream>>>(in, weight, tmp, M, N, K);
        cudaMemcpyAsync(out, tmp, M*N*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaFree(tmp);
    } else {
        linear_fp32_kernel<<<grid, block, 0, stream>>>(in, weight, out, M, N, K);
    }
}

// Linear backward via cuBLAS
// forward:  out[M,N] = in[M,K] @ weight[N,K]^T
// d_in[M,K] = d_out[M,N] @ weight[N,K]
// dw[N,K] = d_out[M,N]^T @ in[M,K]
//
// cuBLAS column-major:
//   d_in:  cublasSgemm(N, N, K, M, N, 1.0, weight, K, dout, N, 0.0, d_in, K)
//   dw:    cublasSgemm(N, T, K, N, M, 1.0, in, K, dout, N, 0.0, dw, K)
void launch_linear_bwd_fp32(const float* dout, const float* in,
    const float* weight, float* d_in, float* dw,
    int M, int N, int K, cudaStream_t stream) {
    cublasHandle_t h = get_cublas_handle();
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                K, M, N, &alpha, weight, K, dout, N, &beta, d_in, K);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                K, N, M, &alpha, in, K, dout, N, &beta, dw, K);
}

