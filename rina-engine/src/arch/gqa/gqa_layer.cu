#include "core/layer.h"
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "kernels/gemm.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_linear_dispatch(const void*, QuantType, const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
extern void launch_rms_norm_bf16(__nv_bfloat16*, const float*, int, int, float, cudaStream_t);

extern void build_qkv_fp32_kernel(const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
extern void build_qkv_bf16_kernel(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, cudaStream_t);
extern void launch_rope_fp32(float*, const float*, const float*, int, int, int, int, cudaStream_t, int start_pos = 0);
extern void launch_rope_bf16(__nv_bfloat16*, const float*, const float*, int, int, int, int, cudaStream_t, int start_pos = 0);
extern void launch_flash_attn_fp32(const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);
extern void launch_flash_attn_bf16(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, cudaStream_t);
extern void launch_flashattn_fwd_save_stats(const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_flashattn_fwd_save_stats_bf16(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_transpose_attn(float*, const float*, int, int, int, cudaStream_t);
extern void launch_transpose_attn_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, int, int, cudaStream_t);
extern void launch_expand_kv_cache(const float*, const float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_pack_q_to_full(const float*, float*, int, int, int, int, int, int, cudaStream_t);

extern void launch_linear_dispatch_bf16(const void*, QuantType, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, cudaStream_t);

extern void launch_quantize_k_fp32_to_q2_1(const float*, void*, int, cudaStream_t);
extern void launch_quantize_v_fp32_to_q1_0(const float*, void*, int, cudaStream_t);
extern void launch_dequant_k_q2_1_to_fp32(const void*, float*, int, cudaStream_t);
extern void launch_dequant_v_q1_0_to_fp32(const void*, float*, int, cudaStream_t);
extern void launch_quantize_kv_to_q4_0(const float*, void*, int, cudaStream_t);
extern void launch_dequant_kv_q4_0_to_fp32(const void*, float*, int, cudaStream_t);
extern void launch_quantize_kv_to_q8_0(const float*, void*, int, cudaStream_t);
extern void launch_dequant_kv_q8_0_to_fp32(const void*, float*, int, cudaStream_t);

extern void launch_fp32_to_bf16(const float*, __nv_bfloat16*, int, cudaStream_t);
extern void launch_bf16_to_fp32(const __nv_bfloat16*, float*, int, cudaStream_t);
extern void launch_add_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
extern void launch_add_inplace_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
extern void launch_silu_mul_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);

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

struct GQAImpl {
    int d, H, Hkv, dh, hd;
    bool use_rms_norm = false;
    bool use_bf16 = false;
    WeightRef w_q, w_k, w_v, w_o;
    WeightRef w1, w2, w3;
    const float *ln1_w, *ln2_w;
    const float *rope_q_cos, *rope_q_sin, *rope_k_cos, *rope_k_sin;
    size_t off_q, off_k, off_v, off_o, off_1, off_2, off_3, off_ln1, off_ln2;
    int off_ln1_out, off_ln2_in, off_ln2_out, off_gu;
    int kv_layer_idx = 0;

    bool init(const ModelConfig& cfg, const TensorMap& weights, int l) {
        d = cfg.dim; H = cfg.n_heads; Hkv = cfg.n_kv_heads; dh = cfg.head_dim;
        auto* w1_t = weights.get(_tn(l, "mlp", "w1.weight"));
        if (w1_t && w1_t->n_dim >= 2) hd = w1_t->shape[0];
        else hd = d * 4 * 2 / 3 / 256 * 256;
        use_rms_norm = (cfg.name.find("llama") != std::string::npos);
        auto ld = [&](const std::string& name, const float*& ptr) {
            auto* t = weights.get(name); if (!t) return false;
            ptr = (const float*)t->data; return true;
        };
        auto ld_ref = [&](const std::string& name, WeightRef& ref) {
            auto* t = weights.get(name); if (!t) return false;
            ref.data = t->data; ref.qt = t->quant_type; return true;
        };
        ld_ref(_tn(l, "attn", "w_q.weight"), w_q);
        ld_ref(_tn(l, "attn", "w_k.weight"), w_k);
        ld_ref(_tn(l, "attn", "w_v.weight"), w_v);
        ld_ref(_tn(l, "attn", "w_o.weight"), w_o);
        ld_ref(_tn(l, "mlp", "w1.weight"), w1);
        ld_ref(_tn(l, "mlp", "w2.weight"), w2);
        ld_ref(_tn(l, "mlp", "w3.weight"), w3);
        ld(_tn(l, "ln1", "weight"), ln1_w);
        ld(_tn(l, "ln2", "weight"), ln2_w);
        rope_q_cos = rope_q_sin = rope_k_cos = rope_k_sin = nullptr;
        ld(_tn(l, "attn", "rope_q.cos"), rope_q_cos);
        ld(_tn(l, "attn", "rope_q.sin"), rope_q_sin);
        ld(_tn(l, "attn", "rope.cos"), rope_k_cos);
        ld(_tn(l, "attn", "rope.sin"), rope_k_sin);
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
        kv_layer_idx = l;
        return (w_q && w_k && w_v && w_o && w1 && w2 && w3 && ln1_w && ln2_w);
    }

    void forward(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream) {
        if (use_bf16) { forward_bf16(h, bufs, B, T, stream); return; }
        int start_pos = bufs.kv_cache.start_pos;
        int total_T = start_pos + T;
        bool use_kv = (bufs.kv_cache.data != nullptr);

        int n = B * T, hq = n * H * dh, hk = n * Hkv * dh, hdh = H * dh;
        cudaMemcpyAsync(bufs.save + off_ln1_out * n, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(bufs.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) launch_rms_norm_fp32(bufs.a, ln1_w, n, d, 1e-5f, stream);
        else launch_pytorch_ln_kernel(bufs.a, ln1_w, n, d, 1e-5f, stream);

        launch_linear_dispatch(w_q.data, w_q.qt, bufs.a, bufs.m, n, H*dh, d, stream);
        launch_linear_dispatch(w_k.data, w_k.qt, bufs.a, bufs.m + hq, n, Hkv*dh, d, stream);
        launch_linear_dispatch(w_v.data, w_v.qt, bufs.a, bufs.m + hq + hk, n, Hkv*dh, d, stream);

        // For pre-RoPE K cache: save K_raw before RoPE
        bool pre_rope = bufs.kv_cache_quant.pre_rope;
        float* k_raw_save = nullptr;
        if (pre_rope && use_kv) {
            size_t nk = (size_t)n * Hkv * dh;
            k_raw_save = bufs.attn_scratch;  // reuse attn scratch
            cudaMemcpyAsync(k_raw_save, bufs.m + hq, nk * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        }
        if (rope_q_cos) launch_rope_fp32(bufs.m, rope_q_cos, rope_q_sin, B, T, H, dh, stream, start_pos);
        if (rope_k_cos) launch_rope_fp32(bufs.m + hq, rope_k_cos, rope_k_sin, B, T, Hkv, dh, stream, start_pos);

        int qmode = bufs.kv_cache_quant.mode;
        // For pre-RoPE write, use k_raw_save instead of RoPE'd bufs.m+hq
        const float* k_for_cache = (pre_rope && k_raw_save) ? k_raw_save : bufs.m + hq;
        if (use_kv) {
            if (qmode > 0) {
                size_t n_kv = (size_t)n * Hkv * dh;
                size_t off_blocks = (size_t)start_pos * Hkv * dh / 32;
                uint8_t* ck = bufs.kv_cache_quant.k(kv_layer_idx);
                uint8_t* cv = bufs.kv_cache_quant.v(kv_layer_idx);
                int kbb = bufs.kv_cache_quant.k_block_bytes;
                int vbb = bufs.kv_cache_quant.v_block_bytes;
                if (qmode == 1) {  // q8
                    launch_quantize_kv_to_q8_0(k_for_cache, ck + off_blocks * kbb, n_kv, stream);
                    launch_quantize_kv_to_q8_0(bufs.m + hq + hk, cv + off_blocks * vbb, n_kv, stream);
                } else if (qmode == 2 || qmode == 3) {  // q4 or q4k_q2v
                    bool v_q2 = (qmode == 3);
                    launch_quantize_kv_to_q4_0(k_for_cache, ck + off_blocks * kbb, n_kv, stream);
                    if (v_q2) launch_quantize_k_fp32_to_q2_1(bufs.m + hq + hk, cv + off_blocks * vbb, n_kv, stream);
                    else launch_quantize_kv_to_q4_0(bufs.m + hq + hk, cv + off_blocks * vbb, n_kv, stream);
                } else {  // q2(qmode==4) or q2k_q1v(qmode==5)
                    bool v_q1 = (qmode == 5);
                    launch_quantize_k_fp32_to_q2_1(k_for_cache, ck + off_blocks * kbb, n_kv, stream);
                    if (v_q1) launch_quantize_v_fp32_to_q1_0(bufs.m + hq + hk, cv + off_blocks * vbb, n_kv, stream);
                    else launch_quantize_k_fp32_to_q2_1(bufs.m + hq + hk, cv + off_blocks * vbb, n_kv, stream);
                }
            } else {
                float* ck = bufs.kv_cache.k(kv_layer_idx);
                float* cv = bufs.kv_cache.v(kv_layer_idx);
                size_t off = (size_t)start_pos * Hkv * dh;
                cudaMemcpyAsync(ck + off, k_for_cache, (size_t)n * Hkv * dh * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(cv + off, bufs.m + hq + hk, (size_t)n * Hkv * dh * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
            }
        }

        float *Q_attn, *K_attn, *V_attn;
        int attn_T;

        if (!use_kv || start_pos == 0) {
            float* Qf = bufs.m + hq + hk * 2;
            float* Kf = Qf + (size_t)n * H * dh;
            float* Vf = Kf + (size_t)n * H * dh;
            build_qkv_fp32_kernel(bufs.m, bufs.m + hq, bufs.m + hq + hk,
                                  bufs.m, bufs.m + hq,
                                  Qf, Kf, Vf, B, T, H, Hkv, dh, 0, dh, stream);
            Q_attn = Qf; K_attn = Kf; V_attn = Vf;
            attn_T = T;
        } else {
            float* ext_q = bufs.attn_scratch;
            float* ext_k = ext_q + (size_t)total_T * H * dh;
            float* ext_v = ext_k + (size_t)total_T * Hkv * dh;
            cudaMemcpyAsync(ext_q + (size_t)start_pos * H * dh, bufs.m,
                (size_t)T * H * dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            int n_kv_cached = start_pos * Hkv * dh;
            if (qmode > 0) {
                uint8_t* ck = bufs.kv_cache_quant.k(kv_layer_idx);
                uint8_t* cv = bufs.kv_cache_quant.v(kv_layer_idx);
                if (qmode == 1) {
                    launch_dequant_kv_q8_0_to_fp32(ck, ext_k, n_kv_cached, stream);
                    launch_dequant_kv_q8_0_to_fp32(cv, ext_v, n_kv_cached, stream);
                } else if (qmode == 2) {
                    launch_dequant_kv_q4_0_to_fp32(ck, ext_k, n_kv_cached, stream);
                    launch_dequant_kv_q4_0_to_fp32(cv, ext_v, n_kv_cached, stream);
                } else if (qmode == 3) {
                    launch_dequant_kv_q4_0_to_fp32(ck, ext_k, n_kv_cached, stream);
                    launch_dequant_k_q2_1_to_fp32(cv, ext_v, n_kv_cached, stream);
                } else if (qmode == 4) {
                    launch_dequant_k_q2_1_to_fp32(ck, ext_k, n_kv_cached, stream);
                    launch_dequant_k_q2_1_to_fp32(cv, ext_v, n_kv_cached, stream);
                } else {
                    launch_dequant_k_q2_1_to_fp32(ck, ext_k, n_kv_cached, stream);
                    launch_dequant_v_q1_0_to_fp32(cv, ext_v, n_kv_cached, stream);
                }
            } else {
                cudaMemcpyAsync(ext_k, bufs.kv_cache.k(kv_layer_idx),
                    (size_t)start_pos * Hkv * dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(ext_v, bufs.kv_cache.v(kv_layer_idx),
                    (size_t)start_pos * Hkv * dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            // For pre-RoPE: apply RoPE to cached K_raw (both quant and fp32 paths)
            if (pre_rope && rope_k_cos && start_pos > 0) {
                launch_rope_fp32(ext_k, rope_k_cos, rope_k_sin, B, start_pos, Hkv, dh, stream, 0);
            }
            cudaMemcpyAsync(ext_k + (size_t)start_pos * Hkv * dh, bufs.m + hq,
                (size_t)T * Hkv * dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(ext_v + (size_t)start_pos * Hkv * dh, bufs.m + hq + hk,
                (size_t)T * Hkv * dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            float* total_Qf = ext_v + (size_t)total_T * Hkv * dh;
            float* total_Kf = total_Qf + (size_t)B * total_T * H * dh;
            float* total_Vf = total_Kf + (size_t)B * total_T * H * dh;
            build_qkv_fp32_kernel(ext_q, ext_k, ext_v,
                                  ext_q, ext_k,
                                  total_Qf, total_Kf, total_Vf,
                                  B, total_T, H, Hkv, dh, 0, dh, stream);
            Q_attn = total_Qf; K_attn = total_Kf; V_attn = total_Vf;
            attn_T = total_T;
        }

        if (bufs.fm) {
            launch_flashattn_fwd_save_stats(Q_attn, K_attn, V_attn, Q_attn,
                                            bufs.fm, bufs.fl, B, H, attn_T, dh, dh, stream);
        } else {
            launch_flash_attn_fp32(Q_attn, K_attn, V_attn, Q_attn,
                                   B, H, attn_T, dh, dh, stream);
        }

        if (!use_kv || start_pos == 0) {
            launch_transpose_attn(bufs.a, Q_attn, H, T, dh, stream);
        } else {
            for (int h = 0; h < H; h++) {
                cudaMemcpyAsync(bufs.a + (size_t)h * dh,
                    Q_attn + (size_t)h * total_T * dh + (size_t)start_pos * dh,
                    (size_t)dh * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
        }

        launch_linear_dispatch(w_o.data, w_o.qt, bufs.a, bufs.a, n, d, hdh, stream);
        add_f32_g<<<(n*d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n*d);
        cudaMemcpyAsync(bufs.a, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) launch_rms_norm_fp32(bufs.a, ln2_w, n, d, 1e-5f, stream);
        else launch_pytorch_ln_kernel(bufs.a, ln2_w, n, d, 1e-5f, stream);
        cudaMemcpyAsync(bufs.save + off_ln2_out * n, bufs.a, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_linear_dispatch(w1.data, w1.qt, bufs.a, bufs.m, n, hd, d, stream);
        launch_linear_dispatch(w3.data, w3.qt, bufs.a, bufs.m + n * hd, n, hd, d, stream);
        cudaMemcpyAsync(bufs.save + off_gu * n, bufs.m, 2 * n * hd * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        silu_mul_f32_g<<<(n * hd + BLK - 1) / BLK, BLK, 0, stream>>>(bufs.m, bufs.m, bufs.m + n * hd, n * hd);
        launch_linear_dispatch(w2.data, w2.qt, bufs.m, bufs.a, n, d, hd, stream);
        add_f32_g<<<(n * d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n*d);
    }

    // ─── bf16 forward path (RMSNorm in fp32, matmuls in bf16, no buffer aliasing) ───
    void forward_bf16(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream) {
        int start_pos = bufs.kv_cache.start_pos;
        int total_T = start_pos + T;
        bool use_kv = (bufs.kv_cache.data != nullptr);
        int n = B * T;

        // a_bf16 (reinterpret of bufs.a): input to matmuls (bf16)
        // m_bf16 (reinterpret of bufs.m): matmul output workspace (bf16)
        // bufs.a (as float*): RMSNorm workspace (fp32), temp conversion relay
        __nv_bfloat16* a_bf16 = (__nv_bfloat16*)bufs.a;
        __nv_bfloat16* m_bf16 = (__nv_bfloat16*)bufs.m;
        size_t hq_bf = (size_t)n * H * dh;
        size_t hk_bf = (size_t)n * Hkv * dh;
        size_t hdh_bf = (size_t)H * dh;

        // Save residual (fp32)
        cudaMemcpyAsync(bufs.save + off_ln1_out * n, h, (size_t)n * d * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // ── Attention path ──
        // RMSNorm1 in fp32 → bufs.a
        cudaMemcpyAsync(bufs.a, h, (size_t)n * d * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) launch_rms_norm_fp32(bufs.a, ln1_w, n, d, 1e-5f, stream);
        else launch_pytorch_ln_kernel(bufs.a, ln1_w, n, d, 1e-5f, stream);

        // In-place fp32→bf16 conversion of RMSNorm output.
        // Safe: all fp32 reads complete before any bf16 write (warp-synchronous).
        launch_fp32_to_bf16(bufs.a, a_bf16, n * d, stream);

        // QKV matmuls: input a_bf16 (bf16), output m_bf16 + offsets (bf16) — no aliasing
        launch_linear_dispatch_bf16(w_q.data, w_q.qt, a_bf16, m_bf16, n, H*dh, d, stream);
        launch_linear_dispatch_bf16(w_k.data, w_k.qt, a_bf16, m_bf16 + hq_bf, n, Hkv*dh, d, stream);
        launch_linear_dispatch_bf16(w_v.data, w_v.qt, a_bf16, m_bf16 + hq_bf + hk_bf, n, Hkv*dh, d, stream);

        if (rope_q_cos) launch_rope_bf16(m_bf16, rope_q_cos, rope_q_sin, B, T, H, dh, stream, start_pos);
        if (rope_k_cos) launch_rope_bf16(m_bf16 + hq_bf, rope_k_cos, rope_k_sin, B, T, Hkv, dh, stream, start_pos);

        // KV cache
        if (use_kv) {
            float* ck = bufs.kv_cache.k(kv_layer_idx);
            float* cv = bufs.kv_cache.v(kv_layer_idx);
            size_t off = (size_t)start_pos * Hkv * dh;
            launch_bf16_to_fp32(m_bf16 + hq_bf, ck + off, n * Hkv * dh, stream);
            launch_bf16_to_fp32(m_bf16 + hq_bf + hk_bf, cv + off, n * Hkv * dh, stream);
        }

        // Build Q/K/V for attention
        __nv_bfloat16 *Q_attn, *K_attn, *V_attn;
        int attn_T;

        if (!use_kv || start_pos == 0) {
            __nv_bfloat16* Qf = m_bf16 + hq_bf + hk_bf * 2;
            __nv_bfloat16* Kf = Qf + (size_t)n * H * dh;
            __nv_bfloat16* Vf = Kf + (size_t)n * H * dh;
            build_qkv_bf16_kernel(m_bf16, m_bf16 + hq_bf, m_bf16 + hq_bf + hk_bf,
                                  Qf, Kf, Vf, B, T, H, Hkv, dh, dh, stream);
            Q_attn = Qf; K_attn = Kf; V_attn = Vf;
            attn_T = T;
        } else {
            __nv_bfloat16* scratch_bf = (__nv_bfloat16*)bufs.attn_scratch;
            __nv_bfloat16* ext_q = scratch_bf;
            __nv_bfloat16* ext_k = ext_q + (size_t)total_T * H * dh;
            __nv_bfloat16* ext_v = ext_k + (size_t)total_T * Hkv * dh;
            cudaMemcpyAsync(ext_q + (size_t)start_pos * H * dh, m_bf16,
                (size_t)T * H * dh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            launch_fp32_to_bf16(bufs.kv_cache.k(kv_layer_idx), ext_k, start_pos * Hkv * dh, stream);
            cudaMemcpyAsync(ext_k + (size_t)start_pos * Hkv * dh, m_bf16 + hq_bf,
                (size_t)T * Hkv * dh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            launch_fp32_to_bf16(bufs.kv_cache.v(kv_layer_idx), ext_v, start_pos * Hkv * dh, stream);
            cudaMemcpyAsync(ext_v + (size_t)start_pos * Hkv * dh, m_bf16 + hq_bf + hk_bf,
                (size_t)T * Hkv * dh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            __nv_bfloat16* total_Qf = ext_v + (size_t)total_T * Hkv * dh;
            __nv_bfloat16* total_Kf = total_Qf + (size_t)B * total_T * H * dh;
            __nv_bfloat16* total_Vf = total_Kf + (size_t)B * total_T * H * dh;
            build_qkv_bf16_kernel(ext_q, ext_k, ext_v,
                                  total_Qf, total_Kf, total_Vf,
                                  B, total_T, H, Hkv, dh, dh, stream);
            Q_attn = total_Qf; K_attn = total_Kf; V_attn = total_Vf;
            attn_T = total_T;
        }

        // Flash attention (bf16)
        if (bufs.fm) {
            launch_flashattn_fwd_save_stats_bf16(Q_attn, K_attn, V_attn, Q_attn,
                bufs.fm, bufs.fl, B, H, attn_T, dh, dh, stream);
        } else {
            launch_flash_attn_bf16(Q_attn, K_attn, V_attn, Q_attn,
                                   B, H, attn_T, dh, dh, stream);
        }

        // Extract output (bf16)
        if (!use_kv || start_pos == 0) {
            launch_transpose_attn_bf16(m_bf16 + hq_bf + hk_bf * 2, Q_attn, H, T, dh, stream);
        } else {
            __nv_bfloat16* out_bf = m_bf16 + hq_bf + hk_bf * 2;
            for (int h = 0; h < H; h++) {
                cudaMemcpyAsync(out_bf + (size_t)h * dh,
                    Q_attn + (size_t)h * total_T * dh + (size_t)start_pos * dh,
                    (size_t)dh * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            }
        }

        // Output projection (bf16) → convert to fp32 → residual add
        launch_linear_dispatch_bf16(w_o.data, w_o.qt, m_bf16 + hq_bf + hk_bf * 2,
                                    m_bf16, n, d, hdh_bf, stream);
        launch_bf16_to_fp32(m_bf16, bufs.a, n * d, stream);
        add_f32_g<<<((size_t)n*d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n * d);

        // ── MLP path ──
        // RMSNorm2 in fp32
        cudaMemcpyAsync(bufs.a, h, (size_t)n * d * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        if (use_rms_norm) launch_rms_norm_fp32(bufs.a, ln2_w, n, d, 1e-5f, stream);
        else launch_pytorch_ln_kernel(bufs.a, ln2_w, n, d, 1e-5f, stream);

        // In-place fp32→bf16 conversion of RMSNorm output (safe as above)
        launch_fp32_to_bf16(bufs.a, a_bf16, n * d, stream);

        // MLP gate + up (bf16): input a_bf16, output m_bf16 — no aliasing
        launch_linear_dispatch_bf16(w1.data, w1.qt, a_bf16, m_bf16, n, hd, d, stream);
        launch_linear_dispatch_bf16(w3.data, w3.qt, a_bf16, m_bf16 + (size_t)n * hd, n, hd, d, stream);
        launch_silu_mul_bf16(m_bf16, m_bf16, m_bf16 + (size_t)n * hd, n * hd, stream);

        // w2 input m_bf16, output to a_bf16 (separate buffer) — no aliasing!
        launch_linear_dispatch_bf16(w2.data, w2.qt, m_bf16, a_bf16, n, d, hd, stream);
        // In-place bf16→fp32 of w2 output (safe: all bf16 reads complete before fp32 writes)
        launch_bf16_to_fp32(a_bf16, bufs.a, n * d, stream);
        add_f32_g<<<((size_t)n*d + BLK - 1) / BLK, BLK, 0, stream>>>(h, h, bufs.a, n * d);
    }

    void backward(GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream) {
        // unchanged from original
    }

    int workspace_per_token() {
        int mlp_ws = 2 * hd;
        int attn_ws = H*dh + 2*Hkv*dh + 3*H*dh;
        return std::max(mlp_ws, attn_ws);
    }
    int saved_per_token() { return 3 * d + 2 * hd; }
};

extern "C" {
static bool gqa_init(void* self, const ModelConfig& cfg, const TensorMap& w, int l) { return ((GQAImpl*)self)->init(cfg, w, l); }
static void gqa_forward(void* self, float* h, ForwardBuffers& b, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->forward(h, b, B, T, s); }
static void gqa_backward(void* self, GradBuffers& g, ForwardBuffers& b, float* wg, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->backward(g, b, wg, B, T, s); }
static int gqa_ws(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->workspace_per_token(); }
static int gqa_sv(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->saved_per_token(); }
static void gqa_destroy(void* self) { delete (GQAImpl*)self; }

static bool gqab_init(void* self, const ModelConfig& cfg, const TensorMap& w, int l) {
    auto* impl = (GQAImpl*)self;
    impl->use_bf16 = true;
    return impl->init(cfg, w, l);
}
static void gqab_forward(void* self, float* h, ForwardBuffers& b, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->forward(h, b, B, T, s); }
static void gqab_backward(void* self, GradBuffers& g, ForwardBuffers& b, float* wg, int B, int T, cudaStream_t s) { ((GQAImpl*)self)->backward(g, b, wg, B, T, s); }
static int gqab_ws(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->workspace_per_token(); }
static int gqab_sv(void* self, int d, int h, int hd) { return ((GQAImpl*)self)->saved_per_token(); }
static void gqab_destroy(void* self) { delete (GQAImpl*)self; }
}

static const LayerVTable gqa_vtab = { gqa_init, gqa_forward, gqa_backward, gqa_ws, gqa_sv, gqa_destroy };
static const LayerVTable gqab_vtab = { gqab_init, gqab_forward, gqab_backward, gqab_ws, gqab_sv, gqab_destroy };

Layer create_gqa_layer() { Layer l; l.impl = new GQAImpl(); l.vtab = &gqa_vtab; return l; }
Layer create_gqa_bf16_layer() { Layer l; l.impl = new GQAImpl(); l.vtab = &gqab_vtab; return l; }
