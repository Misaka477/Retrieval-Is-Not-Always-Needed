#include "core/quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ——— fp16 → Q2_1 （2-bit + fp16 scale per 32） ———
// 用于 K cache 量化
__global__ void quantize_q2_1_kernel(
    const half* __restrict__ input,
    block_q2_1* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;

    float block[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float v = __half2float(input[idx * 32 + i]);
        block[i] = v;
        float a = fabsf(v);
        if (a > amax) amax = a;
    }

    half scale = __float2half(amax);
    output[idx].scale = scale;

    uint8_t data[8] = {0};
    if (amax > 1e-10f) {
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(block[i] / amax);  // maps to -1, 0, 1
            if (q < -1) q = -1;
            if (q > 1)  q = 1;
            q = q + 1;  // maps to 0, 1, 2
            int byte_idx = i / 4;
            int shift    = (i % 4) * 2;
            data[byte_idx] |= (q & 0x3) << shift;
        }
    }
    memcpy(output[idx].data, data, 8);
}

// ——— fp16 → Q1_0 （1-bit + fp16 scale per 32） ———
// 用于 V cache 量化
__global__ void quantize_q1_0_kernel(
    const half* __restrict__ input,
    block_q1_0* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;

    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(__half2float(input[idx * 32 + i]));
        if (a > amax) amax = a;
    }

    output[idx].scale = __float2half(amax);

    uint32_t bits = 0;
    if (amax > 1e-10f) {
        for (int i = 0; i < 32; i++) {
            float v = __half2float(input[idx * 32 + i]);
            if (v > 0.0f) bits |= (1u << i);
        }
    }
    output[idx].bits = bits;
}

// ——— LSC_Q4 （log-space cumsum 中间量量化） ———
// global max 量化，t 是 shape 为 [B, T, ...] 的 tensor
__global__ void quantize_lsc_q4_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,  // 量化后的值 (近似), scale 存额外位置
    float* __restrict__ scale_out,
    int n
) {
    // find max
    extern __shared__ float s_max[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float a = fabsf(__half2float(input[bid * n + i]));
        if (a > val) val = a;
    }
    s_max[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    float amax = s_max[0];
    float scale = (amax > 1e-10f) ? amax / 7.0f : 1.0f;
    if (tid == 0) scale_out[bid] = scale;
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(input[bid * n + i]);
        int q = (int)roundf(v / scale);
        if (q < -7) q = -7;
        if (q > 7)  q = 7;
        // 直接存储反量化后的 half 近似值（LSC_Q4 是运行时中间量，不持久化）
        output[bid * n + i] = __float2half(q * scale);
    }
}
