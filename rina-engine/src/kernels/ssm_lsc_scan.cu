#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// ——— SSM log-space scan (matching PyTorch _ssm_scan) ———
// mem: [B, T, H, dh], decay: [B, T, H, 1]
// output: [B, T, H, dh]

__global__ void ssm_lsc_scan_kernel(
    const half* __restrict__ mem,
    const half* __restrict__ decay,
    half* __restrict__ sf_out,
    int B, int T, int H, int dh
) {
    int bh = blockIdx.x;
    int d  = threadIdx.x;
    if (bh >= B * H || d >= dh) return;
    int b = bh / H, h = bh % H;

    float log_cs = 0.0f;
    float weighted_cs = 0.0f;

    for (int t = 0; t < T; t++) {
        int idx = ((b * T + t) * H + h);
        float d_val = __half2float(decay[idx]);
        float m_val = __half2float(mem[idx * dh + d]);

        // log-space cumsum matching PyTorch: no upper clamping before log
        if (d_val <= 1e-38f) d_val = 1e-38f;
        log_cs += logf(d_val);
        float ca = expf(log_cs);
        float ca_safe = fmaxf(ca, 1e-8f);
        weighted_cs += m_val / ca_safe;
        float sf = ca * weighted_cs;

        // 防止 0 × inf = NaN
        if (isnan(sf) || isinf(sf)) sf = 0.0f;

        sf_out[idx * dh + d] = __float2half(sf);
    }
}

void launch_ssm_lsc_scan(
    const half* mem, const half* decay, half* sf_out,
    int B, int T, int H, int dh,
    cudaStream_t stream
) {
    dim3 grid(B * H);
    dim3 block(min(dh, 256));
    ssm_lsc_scan_kernel<<<grid, block, 0, stream>>>(mem, decay, sf_out, B, T, H, dh);
}
