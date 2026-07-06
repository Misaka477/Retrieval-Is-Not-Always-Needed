#include "ops/embedding.h"
#include <cstdio>

// ─── fp32 embedding (row lookup) ───
// Named differently from legacy kernels/embedding.cu to avoid symbol collision
__global__ void embedding_fp32_kernel_v2(const float* weight, const int* idx, float* output, int B, int T, int d) {
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

// ─── Legacy fp32-only wrapper (for backward compatibility) ───
void launch_embedding_fp32(const float* weight, const int* idx, float* output,
                           int B, int T, int d, cudaStream_t stream) {
    launch_embedding((const void*)weight, QuantType::FP32, idx, output, B, T, d, stream);
}

// Embedding backward: d_weight[id] += d_out[token] for each token (training use)
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

// ─── Q4_0 embedding (dequant on the fly) ───
__global__ void embedding_q4_0_kernel(const void* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token];
    if (tid < 0) tid = 0;
    int per = (d + blockDim.y - 1) / blockDim.y;
    int start = threadIdx.y * per;
    int end = min(start + per, d);
    // Q4_0: half scale + 16 bytes data = 18 bytes per 32 values
    const uint8_t* w = (const uint8_t*)weight + (size_t)tid * (d / 32 * 18);
    for (int i = start; i < end; i++) {
        int blk = i / 32;
        int in_blk = i % 32;
        const uint8_t* blk_data = w + (size_t)blk * 18;
        uint16_t scale_h = *(const uint16_t*)blk_data;
        float scale = __half2float(__ushort_as_half(scale_h));
        int q = (blk_data[2 + in_blk / 2] >> ((in_blk & 1) << 2)) & 0xF;
        output[token * d + i] = (float)(q - 7) * scale;
    }
}

// ─── Q2_K embedding (dequant on the fly) ───
// Q2_K: scales[16] + qs[64] + d[2] + dmin[2] = 84 bytes per 256 values
// Dequant per element: dall * (scales[is+group*2] & 0xF) * q - dmin * (scales[is+group*2] >> 4)
__global__ void embedding_q2k_kernel(const void* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token];
    if (tid < 0) tid = 0;
    int per = (d + blockDim.y - 1) / blockDim.y;
    int start = threadIdx.y * per;
    int end = min(start + per, d);
    int n_blocks = (d + 255) / 256;
    const uint8_t* row_start = (const uint8_t*)weight + (size_t)tid * n_blocks * 84;
    for (int i = start; i < end; i++) {
        int blk = i / 256;
        int in_blk = i % 256;
        const uint8_t* blk_data = row_start + (size_t)blk * 84;
        float dall = __half2float(__ushort_as_half(*(const uint16_t*)(blk_data + 80)));
        float dmin = __half2float(__ushort_as_half(*(const uint16_t*)(blk_data + 82)));
        int page = in_blk / 128;
        int pos = in_blk % 128;
        int l = pos % 32;
        int group_of_32 = pos / 32;
        int is = 8 * page + l / 16;
        int scale_idx = is + group_of_32 * 2;
        uint8_t sc = blk_data[scale_idx];
        float scale_val = dall * (sc & 0xF);
        float min_val = dmin * (sc >> 4);
        int byte_idx = page * 32 + l;
        int q = (blk_data[16 + byte_idx] >> (group_of_32 * 2)) & 3;
        output[token * d + i] = scale_val * q - min_val;
    }
}

__global__ void embedding_q4_0f_kernel(const void* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token];
    if (tid < 0) tid = 0;
    int per = (d + blockDim.y - 1) / blockDim.y;
    int start = threadIdx.y * per;
    int end = min(start + per, d);
    // Q4_0F: float scale + 16 bytes data = 20 bytes per 32 values
    const uint8_t* w = (const uint8_t*)weight + (size_t)tid * (d / 32 * 20);
    for (int i = start; i < end; i++) {
        int blk = i / 32;
        int in_blk = i % 32;
        const uint8_t* blk_data = w + (size_t)blk * 20;
        float scale = *(const float*)blk_data;
        int q = (blk_data[4 + in_blk / 2] >> ((in_blk & 1) << 2)) & 0xF;
        output[token * d + i] = (float)(q - 7) * scale;
    }
}

void launch_embedding(
    const void* weight, QuantType quant_type,
    const int* idx, float* output,
    int B, int T, int d, cudaStream_t stream) {

    int tokens = B * T;
    dim3 block(128, 4);
    dim3 grid((tokens + block.x - 1) / block.x);

    switch (quant_type) {
        case QuantType::Q4_0:
            embedding_q4_0_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
            break;
        case QuantType::Q4_0F:
            embedding_q4_0f_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
            break;
        case QuantType::GGML_Q2_K:
            embedding_q2k_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
            break;
        default:
            embedding_fp32_kernel_v2<<<grid, block, 0, stream>>>((const float*)weight, idx, output, B, T, d);
            break;
    }
}
