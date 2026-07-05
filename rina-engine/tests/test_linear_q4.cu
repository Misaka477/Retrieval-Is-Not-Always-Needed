#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>

#include "core/quant.h"

extern cudaError_t dequant_matmul_q4_0(
    const void* weight_ptr, const half* input, half* output,
    int M, int N, int K, cudaStream_t stream);

static void quantize_q4_0_cpu(const float* src, uint8_t* dst, int n) {
    int num_blocks = n / 32;
    for (int b = 0; b < num_blocks; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float v = fabsf(src[b * 32 + i]); if (v > amax) amax = v; }
        float scale_f = (amax > 1e-10f) ? amax / 7.0f : 1.0f;
        half scale_h = __float2half(scale_f);
        memcpy(dst, &scale_h, 2); dst += 2;
        uint8_t packed[16] = {0};
        for (int i = 0; i < 32; i++) {
            float v = src[b * 32 + i];
            int q = (int)roundf(v / (scale_f + 1e-8f));
            if (q < -7) q = -7; if (q > 7) q = 7;
            int stored = q + 7;
            packed[i >> 1] |= (stored & 0xF) << ((i & 1) << 2);
        }
        memcpy(dst, packed, 16); dst += 16;
    }
}

// Dequant q4_0 block → fp32
static float dequant_one(const uint8_t* blk_ptr, int idx) {
    half scale_h; memcpy(&scale_h, blk_ptr, 2);
    float scale = __half2float(scale_h);
    int q = (blk_ptr[2 + (idx >> 1)] >> ((idx & 1) << 2)) & 0xF;
    return (float)(q - 7) * scale;
}

static bool test_q4(int M, int N, int K, float& max_diff, float& avg_diff) {
    std::vector<float> h_in(M * K);
    std::vector<float> h_w(N * K);
    for (int i = 0; i < M * K; i++) h_in[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < N * K; i++) h_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    int num_blocks = N * K / 32;
    std::vector<uint8_t> h_w_q4(num_blocks * (int)sizeof(block_q4_0));
    for (int n = 0; n < N; n++)
        quantize_q4_0_cpu(h_w.data() + n * K, h_w_q4.data() + n * (K / 32) * sizeof(block_q4_0), K);

    // Convert input to fp16 (same as kernel input)
    std::vector<half> h_in_half(M * K);
    for (int i = 0; i < M * K; i++) h_in_half[i] = __float2half(h_in[i]);

    // Reference: fp16 input, fp32 matmul → cast to fp16
    std::vector<half> h_ref(M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                int b = k / 32;
                const uint8_t* blk = h_w_q4.data() + (n * (K / 32) + b) * sizeof(block_q4_0);
                sum += __half2float(h_in_half[m * K + k]) * dequant_one(blk, k % 32);
            }
            h_ref[m * N + n] = __float2half(sum);
        }

    half *d_in, *d_out;
    uint8_t* d_w_q4;
    cudaMalloc(&d_in, M * K * sizeof(half));
    cudaMalloc(&d_out, M * N * sizeof(half));
    cudaMalloc(&d_w_q4, num_blocks * sizeof(block_q4_0));

    cudaMemcpy(d_in, h_in_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_q4, h_w_q4.data(), num_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice);

    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaGetLastError();
    dequant_matmul_q4_0(d_w_q4, d_in, d_out, M, N, K, s);
    cudaError_t e = cudaStreamSynchronize(s);
    printf("  Kernel: %s\n", cudaGetErrorString(e));

    std::vector<half> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    max_diff = 0; avg_diff = 0;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(__half2float(h_out[i]) - __half2float(h_ref[i]));
        if (d > max_diff) max_diff = d;
        avg_diff += d;
    }
    avg_diff /= (M * N);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_w_q4);
    cudaStreamDestroy(s);
    return max_diff < 1e-3f;
}

int main() {
    bool all_pass = true;

    printf("Test 1: M=4, N=160, K=640 (SSM scale)\n");
    float md, ad;
    bool p = test_q4(4, 160, 640, md, ad);
    printf("  Max diff: %.8f  Avg diff: %.8f  %s\n", md, ad, p ? "PASS" : "FAIL");
    all_pass = all_pass && p;

    printf("Test 2: M=2, N=2048, K=2048 (Llama 1B scale)\n");
    p = test_q4(2, 2048, 2048, md, ad);
    printf("  Max diff: %.8f  Avg diff: %.8f  %s\n", md, ad, p ? "PASS" : "FAIL");
    all_pass = all_pass && p;

    printf("Test 3: M=1, N=640, K=640 (Jamba dim scale)\n");
    p = test_q4(1, 640, 640, md, ad);
    printf("  Max diff: %.8f  Avg diff: %.8f  %s\n", md, ad, p ? "PASS" : "FAIL");
    all_pass = all_pass && p;

    printf("Test 4: M=8, N=64, K=4096 (thin-wide)\n");
    p = test_q4(8, 64, 4096, md, ad);
    printf("  Max diff: %.8f  Avg diff: %.8f  %s\n", md, ad, p ? "PASS" : "FAIL");
    all_pass = all_pass && p;

    printf("\n=== %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
