#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);

int main() {
    // Replicate the exact SSM w_dq call: M=3, N=160, K=640
    int M=3, N=160, K=640;
    
    float *in, *w, *out_eng, *out_ref;
    cudaMalloc(&in, M*K*sizeof(float));
    cudaMalloc(&w, N*K*sizeof(float));
    cudaMalloc(&out_eng, M*N*sizeof(float));
    cudaMalloc(&out_ref, M*N*sizeof(float));
    
    // Create random data
    std::vector<float> h_in(M*K), h_w(N*K), h_ref(M*N);
    for (int i = 0; i < M*K; i++) h_in[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < N*K; i++) h_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    
    // Compute reference on CPU: out[m][n] = sum_k in[m][k] * w[n][k]
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += h_in[m*K + k] * h_w[n*K + k];
            h_ref[m*N + n] = sum;
        }
    
    cudaMemcpy(in, h_in.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w, h_w.data(), N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_ref, h_ref.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaStream_t s; cudaStreamCreate(&s);
    cudaGetLastError();
    launch_linear_fp32(in, w, out_eng, M, N, K, s);
    cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    printf("Kernel: %s\n", cudaGetErrorString(e));
    
    std::vector<float> h_eng(M*N);
    cudaMemcpy(h_eng.data(), out_eng, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_diff = 0;
    for (int i = 0; i < M*N; i++) {
        float d = abs(h_eng[i] - h_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Max diff: %.6f\n", max_diff);
    
    cudaFree(in); cudaFree(w); cudaFree(out_eng); cudaFree(out_ref);
    cudaStreamDestroy(s);
    return 0;
}
