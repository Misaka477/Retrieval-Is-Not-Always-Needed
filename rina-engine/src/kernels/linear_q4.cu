#include <cstdio>
#include <cuda_bf16.h>
#include "core/kernel_api.h"
#include "core/quant.h"
#include "kernels/gemm.cuh"

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);

// Global pre-allocated tmp buffer for dequant+matmul (avoids OOM from fragmentation)
float* g_dequant_tmp = nullptr;
size_t g_dequant_tmp_sz = 0;

// ——— Q4_0 block format ———
// block_q4_0 { half scale; uint8_t data[16]; }
// 32 values per block, each value ∈ [-7, 7].
// Stored as unsigned with +7 offset: stored = val + 7 → [0, 14].
// Packed 2 per byte: data[j] = (s[2j+1] << 4) | s[2j].
// Dequant: val = (stored - 7) * scale.

// ─── fp16 variant: out[M,N] = in[M,K] @ weight_q4[N,K]^T ───
__global__ void dequant_matmul_q4_0_fp16_kernel(
    const block_q4_0* __restrict__ weight,
    const half* __restrict__ input,
    half* __restrict__ output,
    int M, int N, int K) {

    int idx = blockIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M || n >= N) return;

    int num_blocks = K / 32;
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        block_q4_0 blk = weight[(size_t)n * num_blocks + b];
        float scale = __half2float(blk.scale);

        for (int i = 0; i < 32; i++) {
            int q = (blk.data[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int k = b * 32 + i;
            sum += ((float)(q - 7) * scale) * __half2float(input[(size_t)m * K + k]);
        }
    }

    output[(size_t)m * N + n] = __float2half(sum);
}

cudaError_t dequant_matmul_q4_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream) {

    dequant_matmul_q4_0_fp16_kernel<<<M * N, dim3(1,1,1), 0, stream>>>(
        (const block_q4_0*)weight_ptr, input, output, M, N, K);
    return cudaGetLastError();
}

// ─── fp32 variant: out[M,N] = in[M,K] @ weight_q4[N,K]^T ───
// Uses 1D grid to handle large N (e.g., vocab_size > 65535).
__global__ void dequant_matmul_q4_0_fp32_kernel(
    const block_q4_0* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N, int K) {

    int idx = blockIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M || n >= N) return;

    int num_blocks = K / 32;
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        block_q4_0 blk = weight[(size_t)n * num_blocks + b];
        float scale = __half2float(blk.scale);

        for (int i = 0; i < 32; i++) {
            int q = (blk.data[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int k = b * 32 + i;
            sum += ((float)(q - 7) * scale) * input[(size_t)m * K + k];
        }
    }

    output[(size_t)m * N + n] = sum;
}

void launch_linear_q4_fp32(
    const void* weight_ptr, const float* input, float* output,
    int M, int N, int K, cudaStream_t stream) {

    int total = M * N;
    dequant_matmul_q4_0_fp32_kernel<<<total, dim3(1,1,1), 0, stream>>>(
        (const block_q4_0*)weight_ptr, input, output, M, N, K);
}

// ─── Q4_0F (float scale) — same 4-bit block format, scale is fp32 ───
__global__ void dequant_matmul_q4_0f_fp32_kernel(
    const block_q4_0_f* __restrict__ weight,
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N, int K) {

    int idx = blockIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M || n >= N) return;

    int num_blocks = K / 32;
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        block_q4_0_f blk = weight[(size_t)n * num_blocks + b];
        float scale = blk.scale;

        for (int i = 0; i < 32; i++) {
            int q = (blk.data[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int k = b * 32 + i;
            sum += ((float)(q - 7) * scale) * input[(size_t)m * K + k];
        }
    }

    output[(size_t)m * N + n] = sum;
}

void launch_linear_q4f_fp32(
    const void* weight_ptr, const float* input, float* output,
    int M, int N, int K, cudaStream_t stream) {

    int total = M * N;
    dequant_matmul_q4_0f_fp32_kernel<<<total, dim3(1,1,1), 0, stream>>>(
        (const block_q4_0_f*)weight_ptr, input, output, M, N, K);
}

// ─── Dispatch: auto-select fp32 or q4 kernel based on quant_type ───
void launch_linear_dispatch(
    const void* weight_data, QuantType quant_type,
    const float* input, float* output,
    int M, int N, int K, cudaStream_t stream) {

    // Use pre-allocated tmp if available (avoids OOM from fragmentation)
    extern float* g_dequant_tmp;
    extern size_t g_dequant_tmp_sz;

    auto get_tmp = [&](int n_elems) -> float* {
        size_t need = (size_t)n_elems * sizeof(float);
        if (g_dequant_tmp && g_dequant_tmp_sz >= need) {
            return g_dequant_tmp;
        }
        float* p;
        cudaMalloc(&p, need);
        return p;
    };
    auto free_tmp = [&](float* p) {
        if (p != g_dequant_tmp) cudaFree(p);
    };

    switch (quant_type) {
        case QuantType::Q4_0:
            launch_linear_q4_fp32(weight_data, input, output, M, N, K, stream);
            return;
        case QuantType::Q4_0F:
            launch_linear_q4f_fp32(weight_data, input, output, M, N, K, stream);
            return;
        case QuantType::GGML_Q2_K:
        case QuantType::GGML_Q3_K:
        case QuantType::GGML_IQ4_NL:
        case QuantType::GGML_Q4_K:
        case QuantType::GGML_Q6_K:
        case QuantType::GGML_IQ4_XS:
        {
            int n_elems = N * K;
            float* tmp = get_tmp(n_elems);
            if (!tmp) {
                fprintf(stderr,"  launch_linear_dispatch OOM: N=%d K=%d req=%dMB\n",
                    N, K, (int)((size_t)n_elems*sizeof(float)/1048576));
                return;
            }
            launch_dequant_ggml_blocks(weight_data, tmp, n_elems, quant_type, stream);
            launch_linear_fp32(input, tmp, output, M, N, K, stream);
            // Only sync when using malloc'd tmp (g_dequant_tmp is never freed)
            if (tmp != g_dequant_tmp) { cudaStreamSynchronize(stream); cudaFree(tmp); }
            return;
        }
        default:
            launch_linear_fp32(input, (const float*)weight_data, output, M, N, K, stream);
            return;
    }
}

// ─── KV cache helpers (shared across GQA/MLA layers) ───

// Expand flat K/V [total_T, Hkv*dh] to packed Kf/Vf [B, H, total_T, dh]
__global__ void expand_kv_cache_kernel(
    const float* cache_k, const float* cache_v,
    float* Kf, float* Vf,
    int B, int H, int Hkv, int dh, int total_T) {
    int bh = blockIdx.x;
    int t = blockIdx.y;
    if (bh >= B * H || t >= total_T) return;
    int h = bh % H;
    int h_kv = h % Hkv;
    int d = threadIdx.x;
    if (d >= dh) return;
    // K/V cache layout: [total_T, Hkv, dh] = [total_T, Hkv*dh]
    // Kf/Vf layout: [B, H, total_T, dh] = [B*H, total_T, dh]
    const float* src_k = cache_k + (size_t)t * Hkv * dh + (size_t)h_kv * dh;
    const float* src_v = cache_v + (size_t)t * Hkv * dh + (size_t)h_kv * dh;
    float* dst_k = Kf + (size_t)bh * total_T * dh + (size_t)t * dh;
    float* dst_v = Vf + (size_t)bh * total_T * dh + (size_t)t * dh;
    dst_k[d] = src_k[d];
    dst_v[d] = src_v[d];
}

void launch_expand_kv_cache(
    const float* cache_k, const float* cache_v,
    float* Kf, float* Vf,
    int B, int H, int Hkv, int dh, int total_T, cudaStream_t stream) {
    dim3 grid(B * H, total_T);
    int threads = dh > 256 ? 256 : dh;
    expand_kv_cache_kernel<<<grid, threads, 0, stream>>>(
        cache_k, cache_v, Kf, Vf, B, H, Hkv, dh, total_T);
}

// Pack post-RoPE Q from flat workspace [B, T, H*dh] into Qf [B, H, total_T, dh] at position start_pos
__global__ void pack_q_to_full_kernel(
    const float* Q_flat, float* Qf,
    int B, int H, int dh, int T, int total_T, int start_pos) {
    int b = blockIdx.x, h = blockIdx.y, t = blockIdx.z;
    if (b >= B || h >= H || t >= T) return;
    int dst_pos = start_pos + t;
    int d = threadIdx.x;
    if (d >= dh) return;
    Qf[((size_t)b * H + h) * total_T * dh + (size_t)dst_pos * dh + d] =
        Q_flat[((size_t)b * T + t) * H * dh + (size_t)h * dh + d];
}

void launch_pack_q_to_full(
    const float* Q_flat, float* Qf,
    int B, int H, int dh, int T, int total_T, int start_pos, cudaStream_t stream) {
    dim3 grid(B, H, T);
    int threads = dh > 256 ? 256 : dh;
    pack_q_to_full_kernel<<<grid, threads, 0, stream>>>(
        Q_flat, Qf, B, H, dh, T, total_T, start_pos);
}

// ════════════════════════════════════════════════════════════════
// GGML quant format GPU dequant kernels (direct upload, no CPU dequant)
// ════════════════════════════════════════════════════════════════

// Helper: half→float
static inline __device__ float ggml_half_to_float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}static inline __device__ void get_scale_min_k4_gpu(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

// GPU batch dequant: Q4_K blocks → fp32
__global__ void dequant_block_q4_K_gpu(const uint8_t* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + b * 144;
    float d = ggml_half_to_float(*(const uint16_t*)(blk));
    float dmin = ggml_half_to_float(*(const uint16_t*)(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    int off = b * 256;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_gpu(is + 0, scales, &sc, &m);
        float d1 = d * sc; float m1 = dmin * m;
        get_scale_min_k4_gpu(is + 1, scales, &sc, &m);
        float d2 = d * sc; float m2 = dmin * m;
        for (int l = threadIdx.x; l < 32; l += blockDim.x) {
            dst[off + j + l]      = d1 * (qs[l] & 0xF) - m1;
            dst[off + j + 32 + l] = d2 * (qs[l] >> 4) - m2;
        }
        qs += 32; is += 2;
    }
}

// GPU batch dequant: Q6_K blocks → fp32
__global__ void dequant_block_q6_K_gpu(const uint8_t* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + b * 210;
    float d = ggml_half_to_float(*(const uint16_t*)(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int off = b * 256;
    for (int nblk = 0; nblk < 256; nblk += 128) {
        for (int l = threadIdx.x; l < 32; l += blockDim.x) {
            int is = l / 16;
            auto ti8 = [](int v) { return v < 128 ? v : v - 256; };
            int q1 = ti8((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = ti8((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = ti8((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = ti8((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            dst[off + nblk + l]      = d * sc[is + 0] * q1;
            dst[off + nblk + l + 32] = d * sc[is + 2] * q2;
            dst[off + nblk + l + 64] = d * sc[is + 4] * q3;
            dst[off + nblk + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32;
    }
}

// GPU batch dequant: IQ4_XS blocks → fp32
__global__ void dequant_block_iq4_xs_gpu(const uint8_t* __restrict__ src, float* __restrict__ dst, int n_blocks) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + b * 136;
    float d = ggml_half_to_float(*(const uint16_t*)(blk));
    uint16_t scales_h = *(const uint16_t*)(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    static const float kvals[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    int off = b * 256;
    for (int ib = 0; ib < 8; ib++) {
        int ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf;
        ls |= ((scales_h >> 2*ib) & 3) << 4;
        float dl = d * (ls - 32);
        for (int j = threadIdx.x; j < 16; j += blockDim.x) {
            dst[off + ib*32 + j]      = dl * kvals[qs[j] & 0xf];
            dst[off + ib*32 + 16 + j] = dl * kvals[qs[j] >> 4];
        }
        qs += 16;
    }
}

// ─── Q2_K GPU dequant ───
// 2.625 bpw, 256 values per block, 84 bytes per block
typedef struct __attribute__((packed)) {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

__global__ void dequant_block_q2_K_gpu(const void* __restrict__ vx, float* __restrict__ yy, int n_blocks) {
    int i = blockIdx.x;
    if (i >= n_blocks) return;
    const block_q2_K* x = (const block_q2_K*)vx;
    int tid = threadIdx.x;
    int n = tid / 32;
    int l = tid - 32 * n;
    int is = 8 * n + l / 16;
    uint8_t q = x[i].qs[32 * n + l];
    float* y = yy + i * 256 + 128 * n;
    float dall = ggml_half_to_float(x[i].d);
    float dmin = ggml_half_to_float(x[i].dmin);
    y[l + 0] = dall * (x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is + 0] >> 4);
    y[l + 32] = dall * (x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is + 2] >> 4);
    y[l + 64] = dall * (x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is + 4] >> 4);
    y[l + 96] = dall * (x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is + 6] >> 4);
}

// ─── Q3_K GPU dequant ───
// 3.4375 bpw, 256 values per block, 110 bytes per block
typedef struct __attribute__((packed)) {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} block_q3_K;
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

__global__ void dequant_block_q3_K_gpu(const void* __restrict__ vx, float* __restrict__ yy, int n_blocks) {
    int i = blockIdx.x;
    if (i >= n_blocks) return;
    const block_q3_K* x = (const block_q3_K*)vx;
    int tid = threadIdx.x;
    int r = tid / 4;
    int tid2 = r / 2;
    int is0 = r % 2;
    int l0 = 16 * is0 + 4 * (tid % 4);
    int n = tid2 / 4;
    int j = tid2 - 4 * n;
    uint8_t m = 1 << (4 * n + j);
    int is = 8 * n + 2 * j + is0;
    int shift = 2 * j;
    int8_t us = is < 4 ? (x[i].scales[is] & 0xF) | (((x[i].scales[is + 8] >> 0) & 3) << 4) :
                is < 8 ? (x[i].scales[is] & 0xF) | (((x[i].scales[is + 4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is - 8] >> 4) | (((x[i].scales[is] >> 4) & 3) << 4) :
                          (x[i].scales[is - 8] >> 4) | (((x[i].scales[is - 4] >> 6) & 3) << 4);
    float d_all = ggml_half_to_float(x[i].d);
    float dl = d_all * (us - 32);
    float* y = yy + i * 256 + 128 * n + 32 * j;
    const uint8_t* q = x[i].qs + 32 * n;
    const uint8_t* hm = x[i].hmask;
    for (int l = l0; l < l0 + 4; l++) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
}

// IQ4_NL GPU dequant
// 32 values per block, 18 bytes per block
typedef struct __attribute__((packed)) {
    uint16_t d;
    uint8_t qs[16];
} block_iq4_nl_gpu;
static_assert(sizeof(block_iq4_nl_gpu) == 18, "block_iq4_nl_gpu must be 18 bytes");

__global__ void dequant_block_iq4_nl_gpu(const void* __restrict__ vx, float* __restrict__ yy, int n_blocks) {
    int i = blockIdx.x;
    if (i >= n_blocks) return;
    const block_iq4_nl_gpu* x = (const block_iq4_nl_gpu*)vx;
    int off = i * 32;
    float d = ggml_half_to_float(x[i].d);
    const uint8_t* qs = x[i].qs;
    for (int j = threadIdx.x; j < 32; j += blockDim.x) {
        int q = (qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        static const float kvals[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        yy[off + j] = d * kvals[q];
    }
}

// Launch functions
void launch_dequant_ggml_blocks(const void* src, float* dst, int n_elems, QuantType qt, cudaStream_t stream) {
    int bs = ggml_block_size(qt);
    int ts = ggml_type_size(qt);
    int n_blocks = (n_elems + bs - 1) / bs;
    switch (qt) {
        case QuantType::GGML_Q4_K:
            dequant_block_q4_K_gpu<<<n_blocks, 32, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        case QuantType::GGML_Q6_K:
            dequant_block_q6_K_gpu<<<n_blocks, 32, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        case QuantType::GGML_IQ4_XS:
            dequant_block_iq4_xs_gpu<<<n_blocks, 32, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        case QuantType::GGML_Q2_K:
            dequant_block_q2_K_gpu<<<n_blocks, 64, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        case QuantType::GGML_Q3_K:
            dequant_block_q3_K_gpu<<<n_blocks, 64, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        case QuantType::GGML_IQ4_NL:
            dequant_block_iq4_nl_gpu<<<n_blocks, 32, 0, stream>>>((const uint8_t*)src, dst, n_blocks);
            break;
        default: break;
    }
}

// ─── Q1_0 dequant matmul (1-bit, 32 per block, 6 bytes/block) ───
// output[M,N] = sum_k input[M,K] * deq(weight[N,K])
__global__ void dequant_matmul_q1_0_kernel(
    const block_q1_0* __restrict__ weight,
    const half* __restrict__ input,
    half* __restrict__ output,
    int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k += 32) {
        float s = __half2float(weight[(size_t)n * (K / 32) + (k / 32)].scale);
        uint32_t bits = weight[(size_t)n * (K / 32) + (k / 32)].bits;
        for (int i = 0; i < 32 && k + i < K; i++) {
            float w = (bits & (1u << i)) ? s : -s;
            sum += __half2float(input[(size_t)m * K + k + i]) * w;
        }
    }
    output[(size_t)m * N + n] = __float2half(sum);
}

cudaError_t dequant_matmul_q1_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dequant_matmul_q1_0_kernel<<<grid, block, 0, stream>>>(
        (const block_q1_0*)weight_ptr, input, output, M, N, K);
    return cudaGetLastError();
}

// ─── Q2_1 dequant matmul (2-bit, 32 per block, 10 bytes/block) ───
__global__ void dequant_matmul_q2_1_kernel(
    const block_q2_1* __restrict__ weight,
    const half* __restrict__ input,
    half* __restrict__ output,
    int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k += 32) {
        int blk = (int)((size_t)n * (K / 32) + (k / 32));
        float s = __half2float(weight[blk].scale);
        for (int i = 0; i < 32 && k + i < K; i++) {
            int byte_idx = i / 4, shift = (i % 4) * 2;
            int q = (weight[blk].data[byte_idx] >> shift) & 0x3;
            float w = (float)(q - 1) * s;
            sum += __half2float(input[(size_t)m * K + k + i]) * w;
        }
    }
    output[(size_t)m * N + n] = __float2half(sum);
}

cudaError_t dequant_matmul_q2_1(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dequant_matmul_q2_1_kernel<<<grid, block, 0, stream>>>(
        (const block_q2_1*)weight_ptr, input, output, M, N, K);
    return cudaGetLastError();
}

// ════════════════════════════════════════════════════════════════
// bf16 linear dispatch — input/output as bf16, weights fp32/q4
// ════════════════════════════════════════════════════════════════

// fp32 weight × bf16 input → bf16 output
__global__ void linear_fp32_bf16_kernel(
    const float* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int N, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += __bfloat162float(input[(size_t)m * K + k]) * weight[(size_t)n * K + k];
    output[(size_t)m * N + n] = __float2bfloat16(sum);
}

void launch_linear_fp32_bf16(
    const float* weight, const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    linear_fp32_bf16_kernel<<<grid, block, 0, stream>>>(weight, input, output, M, N, K);
}

// Q4_0 weight × bf16 input → bf16 output
__global__ void dequant_matmul_q4_0_bf16_kernel(
    const block_q4_0* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int N, int K) {
    int idx = blockIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M || n >= N) return;
    int num_blocks = K / 32;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        block_q4_0 blk = weight[(size_t)n * num_blocks + b];
        float scale = __half2float(blk.scale);
        for (int i = 0; i < 32; i++) {
            int q = (blk.data[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int k = b * 32 + i;
            sum += ((float)(q - 7) * scale) * __bfloat162float(input[(size_t)m * K + k]);
        }
    }
    output[(size_t)m * N + n] = __float2bfloat16(sum);
}

void launch_linear_q4_bf16(
    const void* weight_ptr, const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream) {
    dequant_matmul_q4_0_bf16_kernel<<<M * N, dim3(1, 1, 1), 0, stream>>>(
        (const block_q4_0*)weight_ptr, input, output, M, N, K);
}

// Q4_0F weight × bf16 input → bf16 output
__global__ void dequant_matmul_q4_0f_bf16_kernel(
    const block_q4_0_f* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int M, int N, int K) {
    int idx = blockIdx.x;
    int m = idx / N;
    int n = idx % N;
    if (m >= M || n >= N) return;
    int num_blocks = K / 32;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        block_q4_0_f blk = weight[(size_t)n * num_blocks + b];
        float scale = blk.scale;
        for (int i = 0; i < 32; i++) {
            int q = (blk.data[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int k = b * 32 + i;
            sum += ((float)(q - 7) * scale) * __bfloat162float(input[(size_t)m * K + k]);
        }
    }
    output[(size_t)m * N + n] = __float2bfloat16(sum);
}

void launch_linear_q4f_bf16(
    const void* weight_ptr, const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream) {
    dequant_matmul_q4_0f_bf16_kernel<<<M * N, dim3(1, 1, 1), 0, stream>>>(
        (const block_q4_0_f*)weight_ptr, input, output, M, N, K);
}

// bf16 dispatch: auto-select kernel based on quant_type
void launch_linear_dispatch_bf16(
    const void* weight_data, QuantType quant_type,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, int K, cudaStream_t stream) {
    switch (quant_type) {
        case QuantType::Q4_0:
            launch_linear_q4_bf16(weight_data, input, output, M, N, K, stream);
            return;
        case QuantType::Q4_0F:
            launch_linear_q4f_bf16(weight_data, input, output, M, N, K, stream);
            return;
        default:
            launch_linear_fp32_bf16((const float*)weight_data, input, output, M, N, K, stream);
            return;
    }
}
