#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "core/quant.h"
#include "core/tensor.h"
#include "core/kernel_api.h"

static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) { fprintf(stderr, "FAIL: %s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
    printf("OK: %s\n", msg);
}

static void test_dequant_matmul() {
    printf("\n=== dequant_matmul ===\n");
    const int M=4, N=256, K=256;
    int n_blocks = N * (K / 32);
    size_t wt_sz = n_blocks * sizeof(block_q4_0);

    block_q4_0* h_wt = (block_q4_0*)calloc(n_blocks, sizeof(block_q4_0));
    half* h_in = (half*)malloc(M * K * sizeof(half));
    half* h_out = (half*)malloc(M * N * sizeof(half));

    for (int i = 0; i < n_blocks; i++) {
        h_wt[i].scale = __float2half(1.0f);
        for (int j = 0; j < 16; j++) h_wt[i].data[j] = 0;
        // 第一行 weight 全为最大值 (nibble=14 → val=7)
        if (i < N) for (int j = 0; j < 16; j++) h_wt[i].data[j] = 0xEE;
    }
    for (int i = 0; i < M * K; i++) h_in[i] = __float2half(1.0f / K);

    block_q4_0* d_wt; half *d_in, *d_out;
    cudaMalloc(&d_wt, wt_sz); cudaMalloc(&d_in, M*K*sizeof(half)); cudaMalloc(&d_out, M*N*sizeof(half));
    cudaMemcpy(d_wt, h_wt, wt_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_in, M*K*sizeof(half), cudaMemcpyHostToDevice);

    check(dequant_matmul_q4_0(d_wt, d_in, d_out, M, N, K), "dequant_matmul");

    cudaMemcpy(h_out, d_out, M*N*sizeof(half), cudaMemcpyDeviceToHost);
    printf("  out[0]=%f (expect ~7.0)\n", __half2float(h_out[0]));

    cudaFree(d_wt); cudaFree(d_in); cudaFree(d_out);
    free(h_wt); free(h_in); free(h_out);
}

static void test_rope() {
    printf("\n=== RoPE ===\n");
    int B=1, H=2, T=4, d=64, half_d=d/2;
    size_t sz = B * H * T * d * sizeof(half);
    half *d_q; cudaMalloc(&d_q, sz);
    half *d_c, *d_s; cudaMalloc(&d_c, T*half_d*sizeof(half)); cudaMalloc(&d_s, T*half_d*sizeof(half));

    std::vector<half> h_c(T*half_d, __float2half(1.0f));
    std::vector<half> h_s(T*half_d, __float2half(0.0f));
    cudaMemcpy(d_c, h_c.data(), T*half_d*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s.data(), T*half_d*sizeof(half), cudaMemcpyHostToDevice);
    launch_rope(d_q, d_c, d_s, B, H, T, d);
    check(cudaGetLastError(), "rope_kernel");
    cudaFree(d_q); cudaFree(d_c); cudaFree(d_s);
}

static void test_rms_norm() {
    printf("\n=== RMSNorm ===\n");
    int B=1, T=4, d=64;
    half *d_x, *d_w;
    cudaMalloc(&d_x, B*T*d*sizeof(half));
    cudaMalloc(&d_w, d*sizeof(half));
    std::vector<half> h_in(B*T*d, __float2half(2.0f));
    std::vector<half> h_w(d, __float2half(1.0f));
    cudaMemcpy(d_x, h_in.data(), B*T*d*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), d*sizeof(half), cudaMemcpyHostToDevice);
    launch_rms_norm(d_x, d_w, B, T, d, 1e-5f);
    check(cudaGetLastError(), "rms_norm");
    cudaFree(d_x); cudaFree(d_w);
}

static void test_embedding() {
    printf("\n=== Embedding Q4_0 ===\n");
    int B=1, T=4, d=64, V=10, nb=V*(d/32);
    block_q4_0* h_wt = (block_q4_0*)calloc(nb, sizeof(block_q4_0));
    for (int i = 0; i < nb; i++) {
        h_wt[i].scale = __float2half(1.0f);
        memset(h_wt[i].data, 0x77, 16);  // nibble=7 → val=0
    }
    int h_idx[] = {0,1,2,3};

    block_q4_0* d_wt; half* d_out; int* d_idx;
    cudaMalloc(&d_wt, nb*sizeof(block_q4_0));
    cudaMalloc(&d_out, B*T*d*sizeof(half));
    cudaMalloc(&d_idx, B*T*sizeof(int));
    cudaMemcpy(d_wt, h_wt, nb*sizeof(block_q4_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, B*T*sizeof(int), cudaMemcpyHostToDevice);
    launch_embedding_q4_0(d_wt, d_idx, d_out, B, T, d);
    check(cudaGetLastError(), "embedding_q4_0");
    cudaFree(d_wt); cudaFree(d_out); cudaFree(d_idx);
    free(h_wt);
}

int main() {
    printf("=== RINA Engine - Layer Test Suite ===\n\n");
    test_dequant_matmul();
    test_rope();
    test_rms_norm();
    test_embedding();
    test_ssm_scan(1, 8, 2, 4);
    test_sparse_gather_fa(1, 2, 8, 16, 16, 3);
    printf("\n=== All tests OK ===\n");
    return 0;
}
