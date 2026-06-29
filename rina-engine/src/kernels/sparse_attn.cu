#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include "gemm.cuh"

__global__ void gather_kv_kernel(const float* k, const float* v, const int* idx, int H, int T, int Ku, int dq, int dh, int pos, float* kg, float* vg) {
    int h = blockIdx.x, ki = threadIdx.x;
    if (h >= H || ki >= Ku) return;
    int sp = idx[pos * Ku + ki];
    sp = min(sp, T - 1);  // clamp to valid range
    for (int d = 0; d < dq; d++) kg[h*Ku*dq + ki*dq + d] = k[h*T*dq + sp*dq + d];
    for (int d = 0; d < dh; d++) vg[h*Ku*dh + ki*dh + d] = v[h*T*dh + sp*dh + d];
}

void launch_sparse_attn(const float* q, const float* k, const float* v, const int* idx,
                        float* out, int B, int H, int T, int Ku, int dq, int dh, cudaStream_t stream) {
    cublasHandle_t hdl = get_cublas_handle();
    cublasSetStream(hdl, stream);
    float inv_sd = 1.0f / sqrtf((float)dq), alpha = 1.0f, beta = 0.0f;

    float *kg, *vg, *sc;
    cudaMalloc(&kg, H * Ku * dq * sizeof(float));
    cudaMalloc(&vg, H * Ku * dh * sizeof(float));
    cudaMalloc(&sc, H * Ku * sizeof(float));

    for (int p = 0; p < T; p++) {
        int nv = min(p + 1, Ku);  // causal: attend to positions 0..p
        gather_kv_kernel<<<H, Ku, 0, stream>>>(k, v, idx, H, T, Ku, dq, dh, p, kg, vg);

        for (int h = 0; h < H; h++) {
            const float* qh = q + h * T * dq + p * dq;
            const float* kh = kg + h * Ku * dq;
            float* sh = sc + h * Ku;
            cublasSgemm(hdl, CUBLAS_OP_T, CUBLAS_OP_N,
                Ku, 1, dq, &inv_sd, kh, dq, qh, dq, &beta, sh, Ku);
        }

        // Causal softmax over nv (first nv elements), zero out the rest
        for (int h = 0; h < H; h++) {
            float* sh = sc + h * Ku;
            float mx = -1e10f;
            for (int i = 0; i < nv; i++) mx = fmaxf(mx, sh[i]);
            float sum = 0;
            for (int i = 0; i < nv; i++) { float e = expf(sh[i] - mx); sh[i] = e; sum += e; }
            float inv = 1.0f / (sum + 1e-10f);
            for (int i = 0; i < nv; i++) sh[i] *= inv;
            for (int i = nv; i < Ku; i++) sh[i] = 0.0f;
        }

        for (int h = 0; h < H; h++) {
            const float* sh = sc + h * Ku;
            const float* vh = vg + h * Ku * dh;
            float* oh = out + p * H * dh + h * dh;
            cublasSgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                dh, 1, Ku, &alpha, vh, dh, sh, Ku, &beta, oh, dh);
        }
    }

    cudaFree(kg); cudaFree(vg); cudaFree(sc);
}
