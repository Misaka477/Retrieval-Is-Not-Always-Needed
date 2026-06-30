#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels/gemm.cuh"

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);

int main() {
    int M=3, N=160, K=640;
    float *in, *w, *out_cus, *out_cublas;
    cudaMalloc(&in, M*K*sizeof(float));
    cudaMalloc(&w, N*K*sizeof(float));
    cudaMalloc(&out_cus, M*N*sizeof(float));
    cudaMalloc(&out_cublas, M*N*sizeof(float));
    
    // Use actual seed for reproducibility
    srand(42);
    std::vector<float> h_in(M*K), h_w(N*K);
    for (int i = 0; i < M*K; i++) h_in[i] = ((float)rand()/RAND_MAX)*2 - 1;
    for (int i = 0; i < N*K; i++) h_w[i] = ((float)rand()/RAND_MAX)*2 - 1;
    
    cudaMemcpy(in, h_in.data(), M*K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, h_w.data(), N*K*4, cudaMemcpyHostToDevice);
    
    cudaStream_t s; cudaStreamCreate(&s);
    cublasHandle_t ch = get_cublas_handle();
    cublasSetStream(ch, s);
    
    // Custom kernel
    launch_linear_fp32(in, w, out_cus, M, N, K, s);
    
    // cuBLAS
    float a=1,b=0;
    cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, w, N, in, K, &b, out_cublas, N);
    
    cudaStreamSynchronize(s);
    
    std::vector<float> cpu_cus(M*N), cpu_cublas(M*N);
    cudaMemcpy(cpu_cus.data(), out_cus, M*N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_cublas.data(), out_cublas, M*N*4, cudaMemcpyDeviceToHost);
    
    // cuBLAS outputs [N,M] col-major = [160,3]. Custom outputs [M,N] row-major = [3,160].
    // Transpose the cuBLAS result to [M,N] for comparison
    // Compute reference on CPU
    std::vector<float> cpu_ref(M*N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += h_in[m*K + k] * h_w[n*K + k];
            cpu_ref[m*N + n] = s;
        }
    
    float max_cus = 0, max_cublas = 0;
    for (int i = 0; i < M*N; i++) {
        float d_cus = abs(cpu_cus[i] - cpu_ref[i]);
        float d_cublas = abs(cpu_cublas[i] - cpu_ref[i]);
        if (d_cus > max_cus) max_cus = d_cus;
        if (d_cublas > max_cublas) max_cublas = d_cublas;
    }
    printf("Custom vs CPU reference: max_diff = %.6f\n", max_cus);
    printf("cuBLAS vs CPU reference: max_diff = %.6f\n", max_cublas);
    
    // Print first few values
    printf("\nFirst 5 elements:\n");
    for (int i = 0; i < 5; i++)
        printf("  [%d]: custom=%f cublas=%f ref=%f\n", i, cpu_cus[i], cpu_cublas[i], cpu_ref[i]);
    
    cudaFree(in); cudaFree(w); cudaFree(out_cus); cudaFree(out_cublas);
    cudaStreamDestroy(s);
    return 0;
}
