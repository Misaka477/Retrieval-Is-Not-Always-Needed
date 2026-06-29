#include <cuda_runtime.h>
#include <cmath>

// LayerNorm backward — simple single-warp implementation
// Each row uses 1 block with 1 warp (32 threads).
// This avoids all cross-warp reduction complexity.
__global__ void layernorm_bwd_kernel(const float* dout, const float* x_saved,
    const float* gamma, float* dx, float* dgamma,
    int D, int N, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* drow = dout + row * D;
    const float* xrow = x_saved + row * D;
    float* dxrow = dx + row * D;

    // 1. Compute sum(x), sum(x²) via intra-warp reduction
    float m = 0.0f, sq = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = xrow[j];
        m += v;
        sq += v * v;
    }
    for (int w = 16; w > 0; w >>= 1) {
        m += __shfl_xor_sync(0xFFFFFFFF, m, w);
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, w);
    }
    m /= D;
    float inv_std = rsqrtf(fmaxf(sq / D - m * m, 0.0f) + eps);

    // 2. Compute sum(dout*gamma) and sum(dout*gamma*x_norm)
    float s_dg = 0.0f, s_dgxn = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float x_n = (xrow[j] - m) * inv_std;
        float dg = drow[j] * gamma[j];
        s_dg += dg;
        s_dgxn += dg * x_n;
    }
    for (int w = 16; w > 0; w >>= 1) {
        s_dg += __shfl_xor_sync(0xFFFFFFFF, s_dg, w);
        s_dgxn += __shfl_xor_sync(0xFFFFFFFF, s_dgxn, w);
    }
    float m_dout = s_dg / D;
    float m_dx = s_dgxn / D;

    // 3. Compute dx and dgamma
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float x_n = (xrow[j] - m) * inv_std;
        dxrow[j] = (drow[j] * gamma[j] - m_dout - x_n * m_dx) * inv_std;
        if (dgamma) atomicAdd(dgamma + j, drow[j] * x_n);
    }
}

void launch_layernorm_bwd_fp32(const float* dout, const float* x_saved,
    const float* gamma, float* dx, float* dgamma,
    int N, int D, cudaStream_t stream) {
    // 1 warp (32 threads) per row — sufficient for any D
    // __shfl_xor_sync reduces across all 32 threads, giving exact per-row sum
    layernorm_bwd_kernel<<<N, 32, 0, stream>>>(
        dout, x_saved, gamma, dx, dgamma, D, N, 1e-5f);
}
