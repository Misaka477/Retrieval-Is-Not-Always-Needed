#include <cuda_runtime.h>
#include <cmath>

static const int BLK = 256;

__global__ void sigmoid_f32_k(float* x, int n) {
    int i = blockIdx.x*BLK + threadIdx.x; if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

void launch_sigmoid_fp32(float* x, int n, cudaStream_t stream) {
    sigmoid_f32_k<<<(n+BLK-1)/BLK, BLK, 0, stream>>>(x, n);
}

__global__ void ssm_agg_f32_k(const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2, float* da, float* ma, int H, int dh, int n_) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n_ * H) return;
    float dd0 = d0[i], dd1 = d1[i], dd2 = d2[i];
    da[i] = dd0 * dd1 * dd2;
    for (int d = 0; d < dh; d++)
        ma[i * dh + d] = m0[i * dh + d] * dd1 * dd2 + m1[i * dh + d] * dd2 + m2[i * dh + d];
}

void launch_ssm_agg_fp32(const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2, float* da, float* ma,
    int H, int dh, int n, cudaStream_t stream) {
    ssm_agg_f32_k<<<(n*H+BLK-1)/BLK, BLK, 0, stream>>>(m0, m1, m2, d0, d1, d2, da, ma, H, dh, n);
}

__global__ void ssm_scan_f32_k(const float* mem, const float* decay, float* sf,
                                int B, int T, int H, int dh) {
    int bh = blockIdx.x, d = threadIdx.x;
    if (bh >= B * H || d >= dh) return;
    int b = bh / H, h = bh % H; float lcs = 0, wcs = 0;
    for (int t = 0; t < T; t++) {
        int idx = ((b * T + t) * H + h);
        float dv = decay[idx]; if (dv <= 1e-38f) dv = 1e-38f;
        lcs += logf(dv); float ca = expf(lcs);
        wcs += mem[idx * dh + d] / fmaxf(ca, 1e-30f);
        sf[idx * dh + d] = ca * wcs;
    }
}

void launch_ssm_scan_fp32(const float* mem, const float* decay, float* sf,
                          int B, int T, int H, int dh, cudaStream_t stream) {
    ssm_scan_f32_k<<<B*H, dh >= 256 ? 256 : (dh >= 128 ? 128 : 64), 0, stream>>>(mem, decay, sf, B, T, H, dh);
}

// SSM scan backward (BPTT)
// Forward:  sf[t] = ca[t] * wcs[t] where ca = cumprod(decay), wcs = cumsum(mem/(ca+eps))
// Backward with recomputed ca[t] values stored in shared memory.
//
// dout[B*T, H*dh], mem[B*T, H*dh], decay[B*T, H], sf[B*T, H*dh]
// → dmem[B*T, H*dh], ddecay[B*T, H]
__global__ void ssm_scan_bwd_k(const float* dout, const float* mem,
    const float* decay, const float* sf,
    float* dmem, float* ddecay,
    int B, int T, int H, int dh) {
    int bh = blockIdx.x, d = threadIdx.x;
    if (bh >= B * H || d >= dh) return;

    extern __shared__ float smem_ca[];
    int b = bh / H, h = bh % H;

    // Phase 1: forward recompute ca[t] = cumprod(decay[0..t]), store in smem
    float ca = 1.0f;
    for (int t = 0; t < T; t++) {
        int didx = (b * T + t) * H + h;
        float dv = decay[didx];
        if (dv <= 1e-38f) dv = 1e-38f;
        ca *= dv;
        if (threadIdx.x == 0) smem_ca[t] = ca;
        __syncthreads();
    }

    // Phase 2: backward pass
    float d_wcs_acc = 0.0f;
    float d_ca_acc = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        float ca_t = smem_ca[t];
        int midx = ((b * T + t) * H + h) * dh + d;
        int didx = (b * T + t) * H + h;

        float df = dout[midx];
        float m_t = mem[midx];

        // wcs[t] = sf[t] / ca[t]
        float wcs_t = sf[midx] / (ca_t + 1e-10f);

        float d_ca = df * wcs_t + d_ca_acc;
        float d_wcs = df * ca_t + d_wcs_acc;

        float inv = 1.0f / (ca_t + 1e-8f);
        dmem[midx] = d_wcs * inv;
        d_ca += -d_wcs * m_t * inv * inv;

        // Propagate through ca[t] = ca[t-1] * decay[t]
        if (t > 0) {
            float ca_prev = ca_t / decay[didx];
            if (decay[didx] <= 1e-38f) ca_prev = ca_t / 1e-38f;
            atomicAdd(&ddecay[didx], d_ca * ca_prev);
            d_ca_acc = d_ca * decay[didx];
        } else {
            atomicAdd(&ddecay[didx], d_ca * 1.0f);
        }

        d_wcs_acc = d_wcs;
        __syncthreads();
    }
}

void launch_ssm_scan_bwd_fp32(const float* dout, const float* mem,
    const float* decay, const float* sf,
    float* dmem, float* ddecay,
    int B, int T, int H, int dh, cudaStream_t stream) {
    int shmem = T * sizeof(float);
    int threads = dh >= 256 ? 256 : (dh >= 128 ? 128 : 64);
    ssm_scan_bwd_k<<<B*H, threads, shmem, stream>>>(dout, mem, decay, sf, dmem, ddecay, B, T, H, dh);
}

// Sigmoid backward: dout *= sigmoid(x) * (1 - sigmoid(x))
__global__ void sigmoid_bwd_k(float* dout, const float* x, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;
    float s = 1.0f / (1.0f + expf(-x[i]));
    dout[i] *= s * (1.0f - s);
}

void launch_sigmoid_bwd_fp32(float* dout, const float* x, int n, cudaStream_t stream) {
    sigmoid_bwd_k<<<(n+BLK-1)/BLK, BLK, 0, stream>>>(dout, x, n);
}

// SSM aggregate backward
// Forward:
//   da[i] = d0[i]*d1[i]*d2[i]
//   ma[i*dh+d] = m0[i*dh+d]*d1[i]*d2[i] + m1[i*dh+d]*d2[i] + m2[i*dh+d]
// Backward:
//   d(m0) = d_ma * d1 * d2
//   d(m1) = d_ma * d2
//   d(m2) = d_ma
//   d(d0) = d_da * d1 * d2
//   d(d1) = d_da * d0 * d2 + sum_d(d_ma * m0 * d2)
//   d(d2) = d_da * d0 * d1 + sum_d(d_ma * (m0*d1 + m1))
__global__ void ssm_agg_bwd_k(const float* d_ma, const float* d_da,
    const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2,
    float* dm0, float* dm1, float* dm2,
    float* dd0, float* dd1, float* dd2,
    int H, int dh, int n_) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n_ * H) return;
    float dd1_sum = d_da[i] * d0[i] * d2[i];
    float dd2_sum = d_da[i] * d0[i] * d1[i];
    for (int d = 0; d < dh; d++) {
        float g = d_ma[i * dh + d];
        dm0[i * dh + d] = g * d1[i] * d2[i];
        dm1[i * dh + d] = g * d2[i];
        dm2[i * dh + d] = g;
        dd1_sum += g * m0[i * dh + d] * d2[i];
        dd2_sum += g * (m0[i * dh + d] * d1[i] + m1[i * dh + d]);
    }
    dd0[i] = d_da[i] * d1[i] * d2[i];
    dd1[i] = dd1_sum;
    dd2[i] = dd2_sum;
}

void launch_ssm_agg_bwd_fp32(const float* d_ma, const float* d_da,
    const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2,
    float* dm0, float* dm1, float* dm2,
    float* dd0, float* dd1, float* dd2,
    int H, int dh, int n, cudaStream_t stream) {
    ssm_agg_bwd_k<<<(n*H+BLK-1)/BLK, BLK, 0, stream>>>(
        d_ma, d_da, m0, m1, m2, d0, d1, d2,
        dm0, dm1, dm2, dd0, dd1, dd2, H, dh, n);
}
// Inline wrappers for backward compat (previously in model_fp32.cu)
void launch_ssm_scan_inline(const float* mem, const float* decay, float* sf,
    int B, int T, int H, int dh, cudaStream_t stream) {
    extern void launch_ssm_scan_fp32(const float*,const float*,float*,int,int,int,int,cudaStream_t);
    launch_ssm_scan_fp32(mem, decay, sf, B, T, H, dh, stream);
}

void launch_ssm_agg_inline(const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2, float* da, float* ma,
    int H, int dh, int n, cudaStream_t stream) {
    extern void launch_ssm_agg_fp32(const float*,const float*,const float*,
        const float*,const float*,const float*,float*,float*,int,int,int,cudaStream_t);
    launch_ssm_agg_fp32(m0,m1,m2,d0,d1,d2,da,ma,H,dh,n,stream);
}

