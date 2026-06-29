#include <cuda_runtime.h>

// Build per-head Q/K/V arrays from raw MLA projections.
// Qf[Bh,T,dq] = qc(a: [n,H,dh]) + qr(m+oq: [n,H,dhr])
// Kf[Bh,T,dq] = kc(m+ok: [n,Hkv,dh]) + kr(m+okr: [n,Hkv,dhr])
// Vf[Bh,T,dh] = v(m+ov: [n,Hkv,dh])
__global__ void build_qkv_fp32_k(const float* a, const float* kc, const float* v,
    const float* qr, const float* kr,
    float* Qf, float* Kf, float* Vf,
    int B, int T, int H, int Hkv, int dh, int dhr, int dq) {
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    int rep = H / Hkv, b = bh / H, h = bh % H, kv = h / rep;
    for (int t = 0; t < T; t++) {
        int base = (b * T + t);
        for (int i = 0; i < dh; i++) {
            Qf[(bh*T+t)*dq + i] = a[base * H * dh + h * dh + i];
            Kf[(bh*T+t)*dq + i] = kc[base * Hkv * dh + kv * dh + i];
            Vf[(bh*T+t)*dh + i] = v[base * Hkv * dh + kv * dh + i];
        }
        for (int i = 0; i < dhr; i++) {
            Qf[(bh*T+t)*dq + dh + i] = qr[base * H * dhr + h * dhr + i];
            Kf[(bh*T+t)*dq + dh + i] = kr[base * Hkv * dhr + kv * dhr + i];
        }
    }
}

void build_qkv_fp32_kernel(const float* a, const float* kc, const float* v,
    const float* qr, const float* kr, float* Qf, float* Kf, float* Vf,
    int B, int T, int H, int Hkv, int dh, int dhr, int dq, cudaStream_t stream) {
    build_qkv_fp32_k<<<1, 32, 0, stream>>>(a, kc, v, qr, kr, Qf, Kf, Vf, B, T, H, Hkv, dh, dhr, dq);
}

// BuildQKV backward: scatter gradients back to inputs
// atomicAdd because multiple Q heads map to the same K/V head
__global__ void build_qkv_bwd_k(const float* dQf, const float* dKf, const float* dVf,
    float* da, float* dkc, float* dv, float* dqr, float* dkr,
    int B, int T, int H, int Hkv, int dh, int dhr, int dq) {
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    int rep = H / Hkv, b = bh / H, h = bh % H, kv = h / rep;
    for (int t = 0; t < T; t++) {
        int base = (b * T + t);
        for (int i = 0; i < dh; i++) {
            float v = dQf[(bh*T+t)*dq + i];
            if (v != 0.0f) atomicAdd(&da[base * H * dh + h * dh + i], v);
            v = dKf[(bh*T+t)*dq + i];
            if (v != 0.0f) atomicAdd(&dkc[base * Hkv * dh + kv * dh + i], v);
            v = dVf[(bh*T+t)*dh + i];
            if (v != 0.0f) atomicAdd(&dv[base * Hkv * dh + kv * dh + i], v);
        }
        for (int i = 0; i < dhr; i++) {
            float v = dQf[(bh*T+t)*dq + dh + i];
            if (v != 0.0f) atomicAdd(&dqr[base * H * dhr + h * dhr + i], v);
            v = dKf[(bh*T+t)*dq + dh + i];
            if (v != 0.0f) atomicAdd(&dkr[base * Hkv * dhr + kv * dhr + i], v);
        }
    }
}

void build_qkv_bwd_kernel(const float* dQf, const float* dKf, const float* dVf,
    float* da, float* dkc, float* dv, float* dqr, float* dkr,
    int B, int T, int H, int Hkv, int dh, int dhr, int dq, cudaStream_t stream) {
    build_qkv_bwd_k<<<1, 32, 0, stream>>>(dQf, dKf, dVf, da, dkc, dv, dqr, dkr,
        B, T, H, Hkv, dh, dhr, dq);
}
