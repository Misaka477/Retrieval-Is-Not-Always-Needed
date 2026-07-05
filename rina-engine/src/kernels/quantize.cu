#include "core/quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>

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

// ——— fp32 → Q2_1 (K cache) ———
__global__ void quantize_f32_to_q2_1_k(
    const float* __restrict__ input,
    block_q2_1* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float block[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float v = input[(size_t)idx * 32 + i];
        block[i] = v;
        float a = fabsf(v);
        if (a > amax) amax = a;
    }
    half scale = __float2half(amax);
    output[idx].scale = scale;
    uint8_t data[8] = {0};
    if (amax > 1e-10f) {
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(block[i] / amax);
            if (q < -1) q = -1;
            if (q > 1)  q = 1;
            q = q + 1;
            int byte_idx = i / 4;
            int shift = (i % 4) * 2;
            data[byte_idx] |= (q & 0x3) << shift;
        }
    }
    memcpy(output[idx].data, data, 8);
}

void launch_quantize_k_fp32_to_q2_1(
    const float* input, void* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    quantize_f32_to_q2_1_k<<<(nb + 255) / 256, 256, 0, stream>>>(input, (block_q2_1*)output, n);
}

// ——— Q2_1 → fp32 (K cache dequant) ———
__global__ void dequant_q2_1_to_f32_k(
    const block_q2_1* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float s = __half2float(input[idx].scale);
    for (int i = 0; i < 32; i++) {
        int byte_idx = i / 4;
        int shift = (i % 4) * 2;
        int q = (input[idx].data[byte_idx] >> shift) & 0x3;
        output[(size_t)idx * 32 + i] = (q - 1) * s;
    }
}

void launch_dequant_k_q2_1_to_fp32(
    const void* input, float* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    dequant_q2_1_to_f32_k<<<(nb + 255) / 256, 256, 0, stream>>>((const block_q2_1*)input, output, n);
}

// ——— fp32 → Q1_0 (V cache) ———
__global__ void quantize_f32_to_q1_0_k(
    const float* __restrict__ input,
    block_q1_0* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(input[(size_t)idx * 32 + i]);
        if (a > amax) amax = a;
    }
    output[idx].scale = __float2half(amax);
    uint32_t bits = 0;
    if (amax > 1e-10f) {
        for (int i = 0; i < 32; i++) {
            float v = input[(size_t)idx * 32 + i];
            if (v > 0.0f) bits |= (1u << i);
        }
    }
    output[idx].bits = bits;
}

void launch_quantize_v_fp32_to_q1_0(
    const float* input, void* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    quantize_f32_to_q1_0_k<<<(nb + 255) / 256, 256, 0, stream>>>(input, (block_q1_0*)output, n);
}

// ——— Q1_0 → fp32 (V cache dequant) ———
__global__ void dequant_q1_0_to_f32_k(
    const block_q1_0* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float s = __half2float(input[idx].scale);
    uint32_t bits = input[idx].bits;
    for (int i = 0; i < 32; i++) {
        float v = (bits & (1u << i)) ? s : -s;
        output[(size_t)idx * 32 + i] = v;
    }
}

void launch_dequant_v_q1_0_to_fp32(
    const void* input, float* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    dequant_q1_0_to_f32_k<<<(nb + 255) / 256, 256, 0, stream>>>((const block_q1_0*)input, output, n);
}

// ——— fp32 → Q4_0 (4-bit KV cache) ———
__global__ void quantize_f32_to_q4_0_k(
    const float* __restrict__ input,
    block_q4_0* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float amax = 0.0f;
    float block[32];
    for (int i = 0; i < 32; i++) {
        block[i] = input[(size_t)idx * 32 + i];
        float a = fabsf(block[i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 7.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    output[idx].scale = __float2half(scale);
    uint8_t data[16] = {0};
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(block[i] / scale);
        if (q < -7) q = -7;
        if (q > 7) q = 7;
        int qu = q + 7;
        int byte_idx = i / 2;
        int shift = (i & 1) * 4;
        data[byte_idx] |= (qu & 0xF) << shift;
    }
    memcpy(output[idx].data, data, 16);
}

void launch_quantize_kv_to_q4_0(
    const float* input, void* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    quantize_f32_to_q4_0_k<<<(nb + 255) / 256, 256, 0, stream>>>(input, (block_q4_0*)output, n);
}

// ——— Q4_0 → fp32 (4-bit KV cache dequant) ———
__global__ void dequant_q4_0_to_f32_k(
    const block_q4_0* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float s = __half2float(input[idx].scale);
    for (int i = 0; i < 32; i++) {
        int byte_idx = i / 2;
        int shift = (i & 1) * 4;
        int q = (input[idx].data[byte_idx] >> shift) & 0xF;
        output[(size_t)idx * 32 + i] = (q - 7) * s;
    }
}

void launch_dequant_kv_q4_0_to_fp32(
    const void* input, float* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    dequant_q4_0_to_f32_k<<<(nb + 255) / 256, 256, 0, stream>>>((const block_q4_0*)input, output, n);
}

// ——— fp32 → Q8_0 (8-bit KV cache) ———
__global__ void quantize_f32_to_q8_0_k(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,  // [nb * 34] = [nb, half + 32*int8]
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(input[(size_t)idx * 32 + i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    // Write fp16 scale
    half h = __float2half(scale);
    memcpy(output + (size_t)idx * 34, &h, 2);
    // Write quantized int8 values
    for (int i = 0; i < 32; i++) {
        float v = input[(size_t)idx * 32 + i];
        int q = (int)roundf(v / scale);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        output[(size_t)idx * 34 + 2 + i] = (uint8_t)(int8_t)q;
    }
}

void launch_quantize_kv_to_q8_0(
    const float* input, void* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    quantize_f32_to_q8_0_k<<<(nb + 255) / 256, 256, 0, stream>>>(input, (uint8_t*)output, n);
}

// ——— Q8_0 → fp32 (8-bit KV cache dequant) ———
__global__ void dequant_q8_0_to_f32_k(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (idx >= nb) return;
    half h; memcpy(&h, input + (size_t)idx * 34, 2);
    float s = __half2float(h);
    for (int i = 0; i < 32; i++) {
        int8_t q = (int8_t)input[(size_t)idx * 34 + 2 + i];
        output[(size_t)idx * 32 + i] = (float)q * s;
    }
}

void launch_dequant_kv_q8_0_to_fp32(
    const void* input, float* output, int n, cudaStream_t stream) {
    int nb = n / 32;
    dequant_q8_0_to_f32_k<<<(nb + 255) / 256, 256, 0, stream>>>((const uint8_t*)input, output, n);
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
