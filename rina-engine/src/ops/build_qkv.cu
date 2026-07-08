#include <cuda_runtime.h>
#include <cuda_bf16.h>

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

static int build_qkv_grid(int B, int H) {
    int n = B * H;
    int threads = n < 128 ? 32 : 128;
    return (n + threads - 1) / threads;
}

void build_qkv_fp32_kernel(const float* a, const float* kc, const float* v,
    const float* qr, const float* kr, float* Qf, float* Kf, float* Vf,
    int B, int T, int H, int Hkv, int dh, int dhr, int dq, cudaStream_t stream) {
    int n_heads = B * H;
    int threads = n_heads;
    if (threads > 256) threads = 256;
    int blocks = (n_heads + threads - 1) / threads;
    build_qkv_fp32_k<<<blocks, threads, 0, stream>>>(a, kc, v, qr, kr, Qf, Kf, Vf, B, T, H, Hkv, dh, dhr, dq);
}

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
    int threads = 128;
    int blocks = (B * H + threads - 1) / threads;
    build_qkv_bwd_k<<<blocks, threads, 0, stream>>>(dQf, dKf, dVf, da, dkc, dv, dqr, dkr,
        B, T, H, Hkv, dh, dhr, dq);
}

__global__ void build_qkv_bf16_k(
    const __nv_bfloat16* q_raw, const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
    __nv_bfloat16* Qf, __nv_bfloat16* Kf, __nv_bfloat16* Vf,
    int B, int T, int H, int Hkv, int dh, int dq) {
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H) return;
    int rep = H / Hkv, b = bh / H, h = bh % H, kv = h / rep;
    int d_eff = dh;
    for (int t = 0; t < T; t++) {
        int base = b * T + t;
        for (int i = 0; i < d_eff; i++) {
            Qf[((size_t)bh * T + t) * dq + i] = q_raw[(size_t)base * H * dh + (size_t)h * dh + i];
            Kf[((size_t)bh * T + t) * dq + i] = k_raw[(size_t)base * Hkv * dh + (size_t)kv * dh + i];
            Vf[((size_t)bh * T + t) * dh + i] = v_raw[(size_t)base * Hkv * dh + (size_t)kv * dh + i];
        }
        for (int i = d_eff; i < dq; i++) {
            Qf[((size_t)bh * T + t) * dq + i] = __float2bfloat16(0.0f);
            Kf[((size_t)bh * T + t) * dq + i] = __float2bfloat16(0.0f);
        }
    }
}

void build_qkv_bf16_kernel(
    const __nv_bfloat16* q_raw, const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
    __nv_bfloat16* Qf, __nv_bfloat16* Kf, __nv_bfloat16* Vf,
    int B, int T, int H, int Hkv, int dh, int dq, cudaStream_t stream) {
    int threads = 128;
    int blocks = (B * H + threads - 1) / threads;
    build_qkv_bf16_k<<<blocks, threads, 0, stream>>>(q_raw, k_raw, v_raw, Qf, Kf, Vf, B, T, H, Hkv, dh, dq);
}
