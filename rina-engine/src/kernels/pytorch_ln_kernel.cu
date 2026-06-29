// PyTorch Welford LN kernel for fp32
#include <cuda_runtime.h>

__global__ void pt_ln_kernel(float* X, const float* gamma, int D, float eps) {
    int tid = threadIdx.x, bid = blockIdx.x;
    float* row = X + bid * D;
    float m = 0, m2 = 0, c = 0;
    for (int i = tid; i < D; i += blockDim.x) { float v = row[i]; c++; float d = v - m; m += d / c; m2 += d * (v - m); }
    for (int s = 16; s > 0; s >>= 1) {
        float om = __shfl_xor_sync(0xFFFFFFFF, m, s), om2 = __shfl_xor_sync(0xFFFFFFFF, m2, s), oc = __shfl_xor_sync(0xFFFFFFFF, c, s);
        float d = om - m, nc = c + oc;
        if (nc > 0) { float nb = oc / nc; m = (c * m + oc * om) / nc; m2 = m2 + om2 + d * d * c * nb; c = nc; }
    }
    float gm = m, gm2 = m2, gc = c;
    if (blockDim.x > 32) {
        extern __shared__ float smem[];
        if (tid % 32 == 0) { int w = tid / 32; smem[w*3] = m; smem[w*3+1] = m2; smem[w*3+2] = c; }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < blockDim.x / 32; w++) {
                float om = smem[w*3], om2 = smem[w*3+1], oc = smem[w*3+2];
                float d = om - gm, nc = gc + oc;
                if (nc > 0) { float nb = oc / nc; gm = (gc * gm + oc * om) / nc; gm2 = gm2 + om2 + d * d * gc * nb; gc = nc; }
            }
            smem[0] = gm; smem[1] = gm2 / D;
        }
        __syncthreads();
        gm = smem[0]; gm2 = smem[1];
    } else {
        gm = __shfl_sync(0xFFFFFFFF, m, 0);
        gm2 = __shfl_sync(0xFFFFFFFF, m2, 0) / D;
    }
    float inv = rsqrtf(gm2 + eps);
    for (int i = tid; i < D; i += blockDim.x) row[i] = (row[i] - gm) * inv * (gamma ? gamma[i] : 1.0f);
}

extern "C" void launch_pytorch_ln_kernel(float* X, const float* gamma, int N, int D, float eps, cudaStream_t stream) {
    int t = D >= 256 ? 256 : (D >= 128 ? 128 : 64);
    int sh = D > 32 ? (t / 32) * 3 * sizeof(float) : 0;
    pt_ln_kernel<<<N, t, sh, stream>>>(X, gamma, D, eps);
}
