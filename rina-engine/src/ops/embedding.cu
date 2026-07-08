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

// Q4_K embedding (4-bit K-quant, 256 elems/block, 144B/block)
// Matches llama.cpp dequantize_row_q4_K
static inline __device__ void q4k_get_scale_min(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

__global__ void embedding_q4k_kernel(const void* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token]; if (tid < 0) tid = 0;
    int n_blocks = d / 256;
    int per = (n_blocks + blockDim.y - 1) / blockDim.y;
    int blk_start = threadIdx.y * per;
    int blk_end = min(blk_start + per, n_blocks);
    const uint8_t* row = (const uint8_t*)weight + (size_t)tid * n_blocks * 144;
    float* out_row = output + (size_t)token * d;
    for (int b = blk_start; b < blk_end; b++) {
        const uint8_t* blk = row + (size_t)b * 144;
        float d_all = __half2float(*(const half*)(blk));
        float dmin  = __half2float(*(const half*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        float* y = out_row + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_get_scale_min(is + 0, scales, &sc, &m);
            float d1 = d_all * sc; float n1 = dmin * m;
            q4k_get_scale_min(is + 1, scales, &sc, &m);
            float d2 = d_all * sc; float n2 = dmin * m;
            for (int l = 0; l < 32; l++) {
                y[j + l]      = d1 * (qs[l] & 0xF) - n1;
                y[j + 32 + l] = d2 * (qs[l] >> 4) - n2;
            }
            qs += 32; is += 2;
        }
    }
}

// Q6_K embedding (6-bit K-quant, 256 elems/block, 210B/block)
// Matches llama.cpp dequantize_row_q6_K
__global__ void embedding_q6k_kernel(const void* weight, const int* idx, float* output, int B, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * T) return;
    int tid = idx[token]; if (tid < 0) tid = 0;
    int n_blocks = d / 256;
    int per = (n_blocks + blockDim.y - 1) / blockDim.y;
    int blk_start = threadIdx.y * per;
    int blk_end = min(blk_start + per, n_blocks);
    const uint8_t* row = (const uint8_t*)weight + (size_t)tid * n_blocks * 210;
    float* out_row = output + (size_t)token * d;
    for (int b = blk_start; b < blk_end; b++) {
        const uint8_t* blk = row + (size_t)b * 210;
        // block_q6_K layout: ql[128] | qh[64] | scales[16] | d[2]
        float d_all = __half2float(*(const half*)(blk + 208));
        float* y = out_row + b * 256;
        // Process in 128-element chunks (2 chunks per block)
        for (int n = 0; n < 256; n += 128) {
            const uint8_t* ql = blk + (n/2);       // ql advances by 64 per 128 elements
            const uint8_t* qh = blk + 128 + (n/4); // qh advances by 32 per 128 elements
            const int8_t* sc = (const int8_t*)(blk + 192) + (n/16); // sc advances by 8
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l +  0] = d_all * sc[is + 0] * q1;
                y[n + l + 32] = d_all * sc[is + 2] * q2;
                y[n + l + 64] = d_all * sc[is + 4] * q3;
                y[n + l + 96] = d_all * sc[is + 6] * q4;
            }
        }
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
        case QuantType::GGML_Q4_K:
            embedding_q4k_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
            break;
        case QuantType::GGML_Q6_K:
            embedding_q6k_kernel<<<grid, block, 0, stream>>>(weight, idx, output, B, T, d);
            break;
        default:
            embedding_fp32_kernel_v2<<<grid, block, 0, stream>>>((const float*)weight, idx, output, B, T, d);
            break;
    }
}
