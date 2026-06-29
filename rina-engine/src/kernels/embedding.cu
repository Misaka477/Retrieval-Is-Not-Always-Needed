#include <cuda_runtime.h>

__global__ void embedding_fp32_kernel(const float* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int b = token / T, t = token % T;
    int tid = idx[b * T + t];
    if (tid < 0) tid = 0;
    int per = (d + blockDim.y - 1) / blockDim.y;
    int start = threadIdx.y * per;
    int end = min(start + per, d);
    for (int i = start; i < end; i++)
        output[token * d + i] = weight[tid * d + i];
}

void launch_embedding_fp32(const float* weight, const int* idx, float* output, int B, int T, int d, cudaStream_t stream) {
    int tokens = B * T;
    dim3 block(128, 4);
    dim3 grid((tokens + block.x - 1) / block.x);
    embedding_fp32_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
}

// Embedding backward: d_weight[id] += d_out[token] for each token
__global__ void embedding_bwd_kernel(const float* dout, const int* idx,
    float* d_weight, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token];
    if (tid < 0) tid = 0;
    int per = (d + blockDim.y - 1) / blockDim.y;
    int start = threadIdx.y * per;
    int end = min(start + per, d);
    for (int i = start; i < end; i++)
        atomicAdd(&d_weight[tid * d + i], dout[token * d + i]);
}

void launch_embedding_bwd_fp32(const float* dout, const int* idx,
    float* d_weight, int B, int T, int d, cudaStream_t stream) {
    int tokens = B * T;
    dim3 block(128, 4);
    dim3 grid((tokens + block.x - 1) / block.x);
    embedding_bwd_kernel<<<grid, block, 0, stream>>>(dout, idx, d_weight, B, T, d);
}
