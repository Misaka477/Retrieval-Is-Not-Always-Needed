#include "core/layer.h"
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "kernels/gemm.cuh"
#include <cstdio>
#include <cmath>
#include <string>

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);

extern void build_qkv_fp32_kernel(const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
extern void build_qkv_bwd_kernel(const float*, const float*, const float*,
    float*, float*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
extern void launch_flash_attn_fp32(const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);
extern void launch_flashattn_fwd_save_stats(const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_flash_attn_bwd_fp32(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_transpose_attn(float*, const float*, int, int, int, cudaStream_t);
extern void launch_rope_fp32(float*, const float*, const float*, int, int, int, int, cudaStream_t);
extern void launch_rope_bwd_fp32(float*, const float*, const float*, const float*, int, int, int, int, cudaStream_t);
extern void launch_silu_mul_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*, const float*,
    const float*, float*, float*, int, int, cudaStream_t);
extern void launch_silu_mul_inline(float*, const float*, const float*, int, cudaStream_t);
extern void launch_copy_f32(float*, const float*, int, cudaStream_t);

static const int BLK = 256;

__global__ void add_f32_g(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x; if (i < n) c[i] = a[i] + b[i];
}
__global__ void silu_mul_f32_g(float* o, const float* g, const float* u, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

static std::string _tn(int l, const char* c, const char* p) {
    char b[128]; snprintf(b, 128, "transformer.h.%d.%s.%s", l, c, p); return b;
}

static size_t wg_offset_of(const TensorMap& weights, const std::string& name) {
    size_t off = 0;
    for (auto& [n_, wt] : weights.tensors) {
        if (wt.quant_type != QuantType::FP32) continue;
        if (n_ == name) return off;
        off += wt.n_elems;
    }
    return (size_t)-1;
}

// ─── Implementation struct (no inheritance) ───
struct GQAImpl {
    int d, H, Hkv, dh, hd;
    bool use_rms_norm = false;  // Llama uses RMSNorm, RINA uses LayerNorm
    const float *w_q, *w_k, *w_v, *w_o;
    const float *w1, *w2, *w3;
    const float *ln1_w, *ln2_w;
    const float *rope_q_cos, *rope_q_sin, *rope_k_cos, *rope_k_sin;
    size_t off_q, off_k, off_v, off_o, off_1, off_2, off_3, off_ln1, off_ln2;
    int off_ln1_out, off_ln2_in, off_ln2_out, off_gu;

    bool init(const ModelConfig& cfg, const TensorMap& weights, int l) {
        d = cfg.dim; H = cfg.n_heads; Hkv = cfg.n_kv_heads; dh = cfg.head_dim;
        // For MLP hidden dim: use actual w1 shape if loaded, else compute from formula
        auto* w1_t = weights.get(_tn(l, "mlp", "w1.weight"));
        if (w1_t && w1_t->n_dim >= 2) {
            hd = w1_t->shape[0];  // actual intermediate_size from weight
        } else {
            hd = d * 4 * 2 / 3 / 256 * 256;
        }
        // Llama models use RMSNorm; RINA models use LayerNorm
        use_rms_norm = (cfg.name.find("llama") != std::string::npos);
        auto ld = [&](const std::string& name, const float*& ptr) {
            auto* t = weights.get(name);
            if (!t) return false;
            ptr = (const float*)t->data;
            return true;
        };
        ld(_tn(l, "attn", "w_q.weight"), w_q);
        ld(_tn(l, "attn", "w_k.weight"), w_k);
        ld(_tn(l, "attn", "w_v.weight"), w_v);
        ld(_tn(l, "attn", "w_o.weight"), w_o);
        ld(_tn(l, "mlp", "w1.weight"), w1);
        ld(_tn(l, "mlp", "w2.weight"), w2);
        ld(_tn(l, "mlp", "w3.weight"), w3);
        ld(_tn(l, "ln1", "weight"), ln1_w);
        ld(_tn(l, "ln2", "weight"), ln2_w);
        // RoPE tables (optional — standard GQA usually has them)
        rope_q_cos = rope_q_sin = rope_k_cos = rope_k_sin = nullptr;
        ld(_tn(l, "attn", "rope_q.cos"), rope_q_cos);
        ld(_tn(l, "attn", "rope_q.sin"), rope_q_sin);
        ld(_tn(l, "attn", "rope.cos"), rope_k_cos);
        ld(_tn(l, "attn", "rope.sin"), rope_k_sin);
        // If no per-layer RoPE, try global positions 0..max_seq_len (common in HuggingFace)
        // Fallback handled in forward by checking if tables exist
        off_q   = wg_offset_of(weights, _tn(l, "attn", "w_q.weight"));
        off_k   = wg_offset_of(weights, _tn(l, "attn", "w_k.weight"));
        off_v   = wg_offset_of(weights, _tn(l, "attn", "w_v.weight"));
        off_o   = wg_offset_of(weights, _tn(l, "attn", "w_o.weight"));
        off_1   = wg_offset_of(weights, _tn(l, "mlp", "w1.weight"));
        off_2   = wg_offset_of(weights, _tn(l, "mlp", "w2.weight"));
        off_3   = wg_offset_of(weights, _tn(l, "mlp", "w3.weight"));
        off_ln1 = wg_offset_of(weights, _tn(l, "ln1", "weight"));
        off_ln2 = wg_offset_of(weights, _tn(l, "ln2", "weight"));
        off_ln1_out = 0; off_ln2_in = d; off_ln2_out = 2*d; off_gu = 3*d;
        return (w_q && w_k && w_v && w_o && w1 && w2 && w3 && ln1_w && ln2_w);
    }

    void forward(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream) {
        int n = B * T, hq = n * H * dh, hk = n * Hkv * dh, hdh = H * dh;
        cudaMemcpyAsync(bufs.save + off_ln1_out * n, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(bufs.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) {
            launch_rms_norm_fp32(bufs.a, ln1_w, n, d, 1e-5f, stream);
        } else {
            launch_pytorch_ln_kernel(bufs.a, ln1_w, n, d, 1e-5f, stream);
        }
        launch_linear_fp32(bufs.a, w_q, bufs.m, n, H*dh, d, stream);
        launch_linear_fp32(bufs.a, w_k, bufs.m + hq, n, Hkv*dh, d, stream);
        launch_linear_fp32(bufs.a, w_v, bufs.m + hq + hk, n, Hkv*dh, d, stream);
        // Apply RoPE to Q and K (standard GQA)
        if (rope_q_cos) launch_rope_fp32(bufs.m, rope_q_cos, rope_q_sin, B, T, H, dh, stream);
        if (rope_k_cos) launch_rope_fp32(bufs.m + hq, rope_k_cos, rope_k_sin, B, T, Hkv, dh, stream);
        float* Qf = bufs.m + hq + hk * 2;
        float* Kf = Qf + n * H * dh;
        float* Vf = Kf + n * H * dh;
        // RoPE already applied in-place on Q/K; build_qkv just packs with dhr=0
        build_qkv_fp32_kernel(bufs.m, bufs.m + hq, bufs.m + hq + hk,
                              bufs.m, bufs.m + hq,
                              Qf, Kf, Vf, B, T, H, Hkv, dh, 0, dh, stream);
        if (bufs.fm) {
            launch_flashattn_fwd_save_stats(Qf, Kf, Vf, Qf, bufs.fm, bufs.fl, B, H, T, dh, dh, stream);
        } else {
            launch_flash_attn_fp32(Qf, Kf, Vf, Qf, B, H, T, dh, dh, stream);
        }
        launch_transpose_attn(bufs.a, Qf, H, T, dh, stream);
        launch_linear_fp32(bufs.a, w_o, bufs.a, n, d, hdh, stream);
        add_f32_g<<<(n*d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n*d);
        cudaMemcpyAsync(bufs.save + off_ln2_in * n, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(bufs.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) {
            launch_rms_norm_fp32(bufs.a, ln2_w, n, d, 1e-5f, stream);
        } else {
            launch_pytorch_ln_kernel(bufs.a, ln2_w, n, d, 1e-5f, stream);
        }
        cudaMemcpyAsync(bufs.save + off_ln2_out * n, bufs.a, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_linear_fp32(bufs.a, w1, bufs.m, n, hd, d, stream);
        launch_linear_fp32(bufs.a, w3, bufs.m + n * hd, n, hd, d, stream);
        cudaMemcpyAsync(bufs.save + off_gu * n, bufs.m, 2 * n * hd * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        silu_mul_f32_g<<<(n * hd + BLK - 1) / BLK, BLK, 0, stream>>>(bufs.m, bufs.m, bufs.m + n * hd, n * hd);
        launch_linear_fp32(bufs.m, w2, bufs.a, n, d, hd, stream);
        add_f32_g<<<(n * d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n*d);
    }

    void backward(GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream) {
        int n = B * T, hq = n * H * dh, hk = n * Hkv * dh, hdh = H * dh;
        cublasHandle_t ch = get_cublas_handle();
        cublasSetStream(ch, stream);
        float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
        const float* sv_ln1_out = bufs.save + off_ln1_out * n;
        const float* sv_ln2_in = bufs.save + off_ln2_in * n;
        const float* sv_ln2_out = bufs.save + off_ln2_out * n;
        const float* sv_gu = bufs.save + off_gu * n;
        // MLP backward
        launch_copy_f32(grad.da, grad.dh, n*d, stream);
        launch_silu_mul_inline(grad.dm, sv_gu, sv_gu + n*hd, n*hd, stream);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, hd, n, d, &a1, w2, hd, grad.da, d, &b0, grad.dm, hd);
        if (off_2 != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, hd, d, n, &a1, grad.dm, hd, grad.da, d, &b1, wg+off_2, hd);
        launch_silu_mul_bwd_fp32(grad.dm, sv_gu, sv_gu+n*hd, grad.dm, grad.dm+n*hd, n*hd, stream);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, d, n, hd, &a1, w1, d, grad.dm, hd, &b1, grad.da, d);
        if (off_1 != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, d, hd, n, &a1, sv_ln2_out, d, grad.dm, hd, &b1, wg+off_1, d);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, d, n, hd, &a1, w3, d, grad.dm+n*hd, hd, &b1, grad.da, d);
        if (off_3 != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, d, hd, n, &a1, sv_ln2_out, d, grad.dm+n*hd, hd, &b1, wg+off_3, d);
        launch_layernorm_bwd_fp32(grad.da, sv_ln2_in, ln2_w, grad.da, 0, n, d, stream);
        add_f32_g<<<(n*d + BLK - 1) / BLK, BLK, 0, stream>>>(grad.dh, grad.dh, grad.da, n*d);
        // Path backward
        launch_copy_f32(grad.da, grad.dh, n*d, stream);
        float* d_attn = grad.dm + hq + hk * 2;
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, hdh, n, d, &a1, w_o, hdh, grad.da, d, &b0, d_attn, hdh);
        if (off_o != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, hdh, d, n, &a1, d_attn, hdh, grad.da, d, &b1, wg+off_o, hdh);
        float* Qf = d_attn, *Kf = Qf + n*H*dh, *Vf = Kf + n*H*dh;
        launch_transpose_attn(Qf, d_attn, H, T, dh, stream);
        if (bufs.fm && bufs.fl) {
            cudaMemsetAsync(Qf, 0, n*H*dh*sizeof(float), stream);
            cudaMemsetAsync(Kf, 0, n*H*dh*sizeof(float), stream);
            cudaMemsetAsync(Vf, 0, n*H*dh*sizeof(float), stream);
            launch_flash_attn_bwd_fp32(Qf, Kf, Vf, Qf, Qf, bufs.fm, bufs.fl, Qf, Kf, Vf, B, H, T, dh, dh, stream);
        }
        build_qkv_bwd_kernel(Qf, Kf, Vf, grad.da, grad.dm+hq, grad.dm+hq+hk, nullptr, nullptr, B, T, H, Hkv, dh, 0, dh, stream);
        // RoPE backward on Q and K (if RoPE was applied in forward)
        if (rope_q_cos) launch_rope_bwd_fp32(grad.da, grad.da, rope_q_cos, rope_q_sin, B, T, H, dh, stream);
        if (rope_k_cos) launch_rope_bwd_fp32(grad.dm+hq, grad.dm+hq, rope_k_cos, rope_k_sin, B, T, Hkv, dh, stream);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, d, n, Hkv*dh, &a1, w_v, d, grad.dm+hq+hk, Hkv*dh, &b1, grad.da, d);
        if (off_v != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, Hkv*dh, d, n, &a1, grad.dm+hq+hk, Hkv*dh, grad.da, d, &b1, wg+off_v, Hkv*dh);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, d, n, Hkv*dh, &a1, w_k, d, grad.dm+hq, Hkv*dh, &b1, grad.da, d);
        if (off_k != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, Hkv*dh, d, n, &a1, grad.dm+hq, Hkv*dh, grad.da, d, &b1, wg+off_k, Hkv*dh);
        cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, d, n, H*dh, &a1, w_q, d, grad.da, H*dh, &b1, grad.da, d);
        if (off_q != (size_t)-1)
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_T, H*dh, d, n, &a1, grad.da, H*dh, grad.da, d, &b1, wg+off_q, H*dh);
        launch_layernorm_bwd_fp32(grad.da, sv_ln1_out, ln1_w, grad.da, 0, n, d, stream);
        add_f32_g<<<(n*d + BLK - 1) / BLK, BLK, 0, stream>>>(grad.dh, grad.dh, grad.da, n*d);
    }

    int workspace_per_token() {
        int hq = H * dh, hk = Hkv * dh;
        int mlp_ws = 2 * hd;  // gate + up before silu_mul
        int attn_ws = hq + 2 * hk + 3 * H * dh;
        return std::max(mlp_ws, attn_ws);
    }

    int saved_per_token() {
        return 3 * d + 2 * hd;
    }
};

// ─── Vtable thunks (cast void* -> GQAImpl*) ───
extern "C" {
static bool gqa_init(void* self, const ModelConfig& cfg, const TensorMap& w, int l) { return ((GQAImpl*)self)->init(cfg, w, l); }
static void gqa_forward(void* self, float* h, ForwardBuffers& b, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->forward(h, b, B, T, s); }
static void gqa_backward(void* self, GradBuffers& g, ForwardBuffers& b, float* wg, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->backward(g, b, wg, B, T, s); }
static int gqa_ws(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->workspace_per_token(); }
static int gqa_sv(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->saved_per_token(); }
static void gqa_destroy(void* self) { delete (GQAImpl*)self; }
}

static const LayerVTable gqa_vtab = {
    gqa_init, gqa_forward, gqa_backward, gqa_ws, gqa_sv, gqa_destroy
};

Layer create_gqa_layer() {
    Layer l;
    l.impl = new GQAImpl();
    l.vtab = &gqa_vtab;
    return l;
}
