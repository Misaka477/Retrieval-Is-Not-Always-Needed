#include "core/layer.h"
#include "core/config.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "core/buffer.h"
#include "ops/saxpy.h"
#include "ops/silu_mul.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern void launch_linear_dispatch(const void*, QuantType, const float*, float*, int, int, int, cudaStream_t);
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
extern void launch_rope_fp32(float*, const float*, const float*, int, int, int, int, cudaStream_t, int start_pos = 0);
extern void launch_flash_attn_fp32(const float*, const float*, const float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_flashattn_fwd_save_stats(const float*, const float*, const float*, float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_transpose_attn(float*, const float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void build_qkv_fp32_kernel(const float*, const float*, const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, int, int, cudaStream_t);

__global__ void expand_kpe_kernel(const float* src, float* dst, int n, int H, int rope_dim) {
    int t = blockIdx.x;
    int i = threadIdx.x;
    if (t >= n || i >= rope_dim) return;
    float val = src[t * rope_dim + i];
    for (int h = 0; h < H; h++) {
        dst[((size_t)t * H + h) * rope_dim + i] = val;
    }
}

static std::string _tn(int l, const char* c, const char* p) {
    char b[128]; snprintf(b, 128, "transformer.h.%d.%s.%s", l, c, p); return b;
}

struct DeepseekMLAImpl {
    int d, H, Hkv, dh, dhr, dq, dc, hd;
    int nope_dim, v_dim;
    bool is_moe = false;
    int kv_layer_idx = 0;

    WeightRef w_q, w_kv_a, w_kv_b, w_o;
    WeightRef w1, w2, w3;
    WeightRef gate_inp, gate_exps, up_exps, down_exps;
    WeightRef gate_shexp, up_shexp, down_shexp;

    int64_t expert_stride_gu;  // bytes per expert for gate/up (Q2_K)
    int64_t expert_stride_dn;  // bytes per expert for down (IQ4_NL)

    const float *ln1_w, *ln2_w;
    const float *k_norm_w;
    const float *rqc, *rqs;

    size_t off_ln1, off_ln2, off_lp, off_l2i, off_l2o, off_gu;

    bool init(const ModelConfig& cfg, const TensorMap& w, int l) {
        d = cfg.dim; H = cfg.n_heads; Hkv = cfg.n_kv_heads;
        dh = cfg.head_dim;
        dhr = cfg.d_h_r > 0 ? cfg.d_h_r : 64;
        dc = cfg.d_c > 0 ? cfg.d_c : 512;
        dq = dh + dhr;
        nope_dim = dh;
        v_dim = dh;
        hd = d * 4 * 2 / 3 / 256 * 256;
        kv_layer_idx = l;

        auto ld = [&](const std::string& n, const float*& p) {
            auto* t = w.get(n); if (!t) return false;
            p = (const float*)t->data; return true;
        };
        auto ld_ref = [&](const std::string& n, WeightRef& ref) {
            auto* t = w.get(n); if (!t) return false;
            ref.data = t->data; ref.qt = t->quant_type; return true;
        };

        ld_ref(_tn(l, "attn", "w_q.weight"), w_q);
        ld_ref(_tn(l, "attn", "w_kv_a.weight"), w_kv_a);
        ld(_tn(l, "attn", "k_norm.weight"), k_norm_w);
        ld_ref(_tn(l, "attn", "w_kv_b.weight"), w_kv_b);
        ld_ref(_tn(l, "attn", "w_o.weight"), w_o);
        ld(_tn(l, "ln1", "weight"), ln1_w);
        ld(_tn(l, "ln2", "weight"), ln2_w);
        ld(_tn(l, "attn", "rope_q.cos"), rqc);
        ld(_tn(l, "attn", "rope_q.sin"), rqs);

        // Compute actual MLP hidden dim from weights
        is_moe = false;
        if (l > 0) {
            auto* gip = w.get(_tn(l, "mlp", "gate_inp.weight"));
            if (gip) {
                is_moe = true;
                ld_ref(_tn(l, "mlp", "gate_inp.weight"), gate_inp);
                ld_ref(_tn(l, "mlp", "gate_exps.weight"), gate_exps);
                ld_ref(_tn(l, "mlp", "up_exps.weight"), up_exps);
                ld_ref(_tn(l, "mlp", "down_exps.weight"), down_exps);
                ld_ref(_tn(l, "mlp", "gate_shexp.weight"), gate_shexp);
                ld_ref(_tn(l, "mlp", "up_shexp.weight"), up_shexp);
                ld_ref(_tn(l, "mlp", "down_shexp.weight"), down_shexp);
                if (gate_shexp) hd = gate_shexp.data ? 2816 : hd;
                if (up_shexp) hd = up_shexp.data ? 2816 : hd;

                // Compute correct byte stride per expert for quantized weights
                int n_elems_per_expert = d * 1408;
                auto stride = [&](QuantType qt, int ne) -> int64_t {
                    if (qt == QuantType::FP32) return (int64_t)ne * sizeof(float);
                    int bs = ggml_block_size(qt);
                    int ts = ggml_type_size(qt);
                    return (int64_t)((ne + bs - 1) / bs) * ts;
                };
                expert_stride_gu = stride(gate_exps.qt, n_elems_per_expert);
                expert_stride_dn = stride(down_exps.qt, n_elems_per_expert);
            }
        }
        if (!is_moe) {
            ld_ref(_tn(l, "mlp", "w1.weight"), w1);
            ld_ref(_tn(l, "mlp", "w2.weight"), w2);
            ld_ref(_tn(l, "mlp", "w3.weight"), w3);
            if (w1 && w1.data && w1.qt == QuantType::FP32) {
                auto* t = w.get(_tn(l, "mlp", "w1.weight"));
                if (t) hd = t->shape[0];
            } else if (w1 && w1.data) {
                auto* t = w.get(_tn(l, "mlp", "w1.weight"));
                if (t) hd = t->shape[0];
            }
        }

        if (is_moe) {
            bool ok = w_q && w_kv_a && k_norm_w && w_kv_b && w_o && ln1_w && ln2_w &&
                  gate_inp && gate_exps && up_exps && down_exps &&
                  gate_shexp && up_shexp && down_shexp;
            if (!ok) {
                fprintf(stderr, "init failed for MoE layer %d: w_q=%d w_kv_a=%d k_norm=%d w_kv_b=%d w_o=%d ln1=%d ln2=%d gate_inp=%d gate_exps=%d up_exps=%d down_exps=%d gate_shexp=%d up_shexp=%d down_shexp=%d\n",
                    l, w_q?1:0, w_kv_a?1:0, k_norm_w?1:0, w_kv_b?1:0, w_o?1:0, ln1_w?1:0, ln2_w?1:0,
                    gate_inp?1:0, gate_exps?1:0, up_exps?1:0, down_exps?1:0,
                    gate_shexp?1:0, up_shexp?1:0, down_shexp?1:0);
                return false;
            }
        } else {
            ld_ref(_tn(l, "mlp", "w1.weight"), w1);
            ld_ref(_tn(l, "mlp", "w2.weight"), w2);
            ld_ref(_tn(l, "mlp", "w3.weight"), w3);
            if (w1 && w1.data) {
                auto* t = w.get(_tn(l, "mlp", "w1.weight"));
                if (t) hd = t->shape[0];
            }
            // Dense MLP w1/w3 dim: for layer 0 it's 10944, which needs large tmp
            // Tile dense matmul to avoid OOM with fragmented GPU memory
            if (!w1 || !w2 || !w3) {
                fprintf(stderr, "init failed for dense layer %d: missing weights\n", l);
                return false;
            }
        }

        off_ln1 = 0; off_lp = d; off_l2i = 2*d; off_l2o = 3*d; off_gu = 4*d;
        return true;
    }

    int ws_per_token() {
        int attn_ws = H * dq + 2 * H * nope_dim + H * v_dim + H * dhr;
        int mlp_ws = 2 * hd;
        if (is_moe) {
            mlp_ws = std::max(mlp_ws, H * 1408 * 2 + H * 2816 * 2);
        }
        return std::max(attn_ws, mlp_ws);
    }
    int sv_per_token() { return d + dc + d + d + d + 2 * hd; }

    void forward(float* h, ForwardBuffers& b, int B, int T, cudaStream_t s) {
        int n = B * T;
        int start_pos = b.mla_kv_cache.start_pos;
        int total_T = start_pos + T;
        bool use_kv = (b.mla_kv_cache.data != nullptr);

        cudaMemcpyAsync(b.save + off_ln1 * n, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(b.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);
        launch_rms_norm_fp32(b.a, ln1_w, n, d, 1e-5f, s);

        float* Q = b.m;
        launch_linear_dispatch(w_q.data, w_q.qt, b.a, Q, n, H * dq, d, s);

        float* c_kv_pe = b.a + n * d;
        launch_linear_dispatch(w_kv_a.data, w_kv_a.qt, b.a, c_kv_pe, n, dc + dhr, d, s);

        float* c_kv = c_kv_pe;
        float* k_pe_raw = c_kv_pe + n * dc;

        launch_rms_norm_fp32(c_kv, k_norm_w, n, dc, 1e-5f, s);

        float* K_nope = b.m + n * H * dq;
        float* V = K_nope + n * H * nope_dim;
        launch_linear_dispatch(w_kv_b.data, w_kv_b.qt, c_kv, K_nope, n, H * (nope_dim + v_dim), dc, s);

        float* k_pe_exp = b.m + n * H * dq + n * H * (nope_dim + v_dim);
        if (dhr > 0) {
            expand_kpe_kernel<<<n, dhr, 0, s>>>(k_pe_raw, k_pe_exp, n, H, dhr);
        }

        float* Q_rope = Q + n * H * nope_dim;
        if (rqc) launch_rope_fp32(Q_rope, rqc, rqs, B, T, H, dhr, s, start_pos);
        if (rqs) launch_rope_fp32(k_pe_exp, rqc, rqs, B, T, H, dhr, s, start_pos);

        float* attn = b.attn_scratch;
        int adq = b.attn_dq > 0 ? b.attn_dq : dq;
        float* Qf = attn;
        float* Kf = Qf + (size_t)B * H * total_T * adq;
        float* Vf = Kf + (size_t)B * H * total_T * adq;

        if (!use_kv || start_pos == 0) {
            build_qkv_fp32_kernel(Q, K_nope, V, Q_rope, k_pe_exp,
                                  Qf, Kf, Vf, B, T, H, Hkv, nope_dim, dhr, dq, s);
            if (b.fm) {
                launch_flashattn_fwd_save_stats(Qf, Kf, Vf, Qf, b.fm, b.fl, B, H, T, dq, nope_dim, s);
            } else {
                launch_flash_attn_fp32(Qf, Kf, Vf, Qf, B, H, T, dq, nope_dim, s);
            }
            launch_transpose_attn(b.a, Qf, H, T, nope_dim, s);
            // Write to KV cache for incremental decode
            if (use_kv) {
                float* cache_k_pe = b.mla_kv_cache.k_pe(kv_layer_idx);
                float* cache_k_nope = b.mla_kv_cache.k_nope(kv_layer_idx);
                float* cache_v = b.mla_kv_cache.v(kv_layer_idx);
                cudaMemcpyAsync(cache_k_pe, k_pe_raw, n * dhr * sizeof(float), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(cache_k_nope, K_nope, n * Hkv * nope_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(cache_v, V, n * Hkv * v_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);
            }
        } else {
            float* ext_q = attn;
            float* ext_k = ext_q + (size_t)total_T * H * adq;
            float* ext_v = ext_k + (size_t)total_T * H * adq;

            cudaMemcpyAsync(ext_q + (size_t)start_pos * H * adq, Q,
                (size_t)T * H * dq * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ext_k + (size_t)start_pos * H * adq, K_nope,
                (size_t)T * H * nope_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ext_v + (size_t)start_pos * H * adq, V,
                (size_t)T * H * v_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);

            float* cache_k_pe = b.mla_kv_cache.k_pe(kv_layer_idx);
            float* cache_k_nope = b.mla_kv_cache.k_nope(kv_layer_idx);
            float* cache_v = b.mla_kv_cache.v(kv_layer_idx);

            launch_rope_fp32(cache_k_pe, rqc, rqs, B, total_T - T, Hkv, dhr, s, 0);
            cudaMemcpyAsync(ext_k, cache_k_nope,
                (size_t)start_pos * Hkv * nope_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ext_v, cache_v,
                (size_t)start_pos * Hkv * v_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);

            float* total_Qf = ext_v + (size_t)total_T * Hkv * v_dim;
            float* total_Kf = total_Qf + (size_t)B * total_T * H * adq;
            float* total_Vf = total_Kf + (size_t)B * total_T * H * adq;

            build_qkv_fp32_kernel(ext_q, ext_k, ext_v, ext_q, ext_k,
                                  total_Qf, total_Kf, total_Vf, B, total_T, H, Hkv, nope_dim, dhr, dq, s);

            if (b.fm) {
                launch_flashattn_fwd_save_stats(total_Qf, total_Kf, total_Vf, total_Qf, b.fm, b.fl, B, H, total_T, dq, nope_dim, s);
            } else {
                launch_flash_attn_fp32(total_Qf, total_Kf, total_Vf, total_Qf, B, H, total_T, dq, nope_dim, s);
            }

            for (int h = 0; h < H; h++) {
                cudaMemcpyAsync(b.a + (size_t)h * nope_dim,
                    total_Qf + (size_t)h * total_T * adq + (size_t)start_pos * adq,
                    (size_t)nope_dim * sizeof(float), cudaMemcpyDeviceToDevice, s);
            }
        }

        launch_linear_dispatch(w_o.data, w_o.qt, b.a, b.a, n, d, H * nope_dim, s);
        launch_add(h, h, b.a, n*d, s);

        cudaMemcpyAsync(b.save + off_l2i * n, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(b.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);
        launch_rms_norm_fp32(b.a, ln2_w, n, d, 1e-5f, s);
        cudaMemcpyAsync(b.save + off_l2o * n, b.a, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);

        if (is_moe) {
            forward_moe(h, b, n, s);
        } else {
            launch_linear_dispatch(w1.data, w1.qt, b.a, b.m, n, hd, d, s);
            launch_linear_dispatch(w3.data, w3.qt, b.a, b.m + n * hd, n, hd, d, s);
            cudaMemcpyAsync(b.save + off_gu * n, b.m, 2 * n * hd * sizeof(float), cudaMemcpyDeviceToDevice, s);
            launch_silu_mul(b.m, b.m, b.m + n * hd, n * hd, s);
            launch_linear_dispatch(w2.data, w2.qt, b.m, b.a, n, d, hd, s);
            launch_add(h, h, b.a, n * d, s);
        }
    }

    void forward_moe(float* h, ForwardBuffers& b, int n, cudaStream_t s) {
        float* shared_gate = b.m;
        float* shared_up = b.m + n * 2816;
        float* shared_out = b.m + n * (2816 + 2816);
        float* h_ln = b.save + off_l2o * n;  // saved ln2 output for router

        launch_linear_dispatch(gate_shexp.data, gate_shexp.qt, b.a, shared_gate, n, 2816, d, s);
        launch_linear_dispatch(up_shexp.data, up_shexp.qt, b.a, shared_up, n, 2816, d, s);
        launch_silu_mul(shared_gate, shared_gate, shared_up, n * 2816, s);
        launch_linear_dispatch(down_shexp.data, down_shexp.qt, shared_gate, shared_out, n, d, 2816, s);
        cudaMemcpyAsync(b.a, shared_out, n * d * sizeof(float), cudaMemcpyDeviceToDevice, s);

        // Router: gate_inp @ h_ln → [n, 64] (use saved ln2 output, not shared output)
        float* router_logits = b.m;
        launch_linear_dispatch(gate_inp.data, gate_inp.qt, h_ln, router_logits, n, 64, d, s);
        cudaStreamSynchronize(s);
        std::vector<float> logits(n * 64);
        cudaMemcpy(logits.data(), router_logits, n * 64 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int t = 0; t < n; t++) {
            float* lp = logits.data() + t * 64;
            float max_lp = *std::max_element(lp, lp + 64);
            float sum_exp = 0;
            for (int e = 0; e < 64; e++) sum_exp += expf(lp[e] - max_lp);
            float inv_sum = 1.0f / sum_exp;

            int topk[6]; float topw[6];
            for (int k = 0; k < 6; k++) { topk[k] = -1; topw[k] = -1e10f; }
            for (int e = 0; e < 64; e++) {
                float w = expf(lp[e] - max_lp) * inv_sum;
                if (w > topw[0]) {
                    topw[0] = w; topk[0] = e;
                    for (int k = 0; k < 5 && topw[k] > topw[k+1]; k++)
                        std::swap(topw[k], topw[k+1]), std::swap(topk[k], topk[k+1]);
                }
            }

            for (int k = 0; k < 6; k++) {
                int e = topk[k]; if (e < 0) continue;
                float prob = topw[k];

                float* eg = b.m;
                float* eu = b.m + 1408;
                float* out_t = b.m + 2816;

                // Use fused Q2_K vec×matmul for M=1 (disabled, needs debug)
                launch_linear_dispatch((const char*)gate_exps.data + e * expert_stride_gu, gate_exps.qt,
                    b.save + off_l2o * n + t * d, eg, 1, 1408, d, s);
                launch_linear_dispatch((const char*)up_exps.data + e * expert_stride_gu, up_exps.qt,
                    b.save + off_l2o * n + t * d, eu, 1, 1408, d, s);
                launch_silu_mul(eg, eg, eu, 1408, s);

                launch_linear_dispatch((const char*)down_exps.data + e * expert_stride_dn, down_exps.qt,
                    eg, out_t, 1, d, 1408, s);

                launch_saxpy(b.a + t * d, out_t, prob, d, s);
            }
        }
        launch_add(h, h, b.a, n * d, s);
    }

    void backward(GradBuffers&, ForwardBuffers&, float*, int, int, cudaStream_t) {}
};

extern "C" {
static bool dsmla_init(void* s, const ModelConfig& c, const TensorMap& w, int l) { return ((DeepseekMLAImpl*)s)->init(c, w, l); }
static void dsmla_fwd(void* s, float* h, ForwardBuffers& b, int B, int T, cudaStream_t st) { ((DeepseekMLAImpl*)s)->forward(h, b, B, T, st); }
static void dsmla_bwd(void*, GradBuffers&, ForwardBuffers&, float*, int, int, cudaStream_t) {}
static int dsmla_ws(void* s, int d, int h, int hd) { return ((DeepseekMLAImpl*)s)->ws_per_token(); }
static int dsmla_sv(void* s, int d, int h, int hd) { return ((DeepseekMLAImpl*)s)->sv_per_token(); }
static void dsmla_del(void* s) { delete (DeepseekMLAImpl*)s; }
}

const LayerVTable dsmla_vtab = { dsmla_init, dsmla_fwd, dsmla_bwd, dsmla_ws, dsmla_sv, dsmla_del };
Layer create_deepseek_mla_dense_layer() { Layer l; l.impl = new DeepseekMLAImpl(); l.vtab = &dsmla_vtab; return l; }
Layer create_deepseek_mla_moe_layer() { Layer l; l.impl = new DeepseekMLAImpl(); ((DeepseekMLAImpl*)l.impl)->is_moe = true; l.vtab = &dsmla_vtab; return l; }
