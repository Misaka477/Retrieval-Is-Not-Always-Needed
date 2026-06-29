#include "core/layer_fp32.h"
#include "core/config.h"
#include "core/tensor.h"
#include <cstdio>
#include <string>

// Kernel declarations (defined in .cu files)
extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_sigmoid_fp32(float*, int, cudaStream_t);
extern void launch_ssm_agg_fp32(const float*, const float*, const float*,
    const float*, const float*, const float*, float*, float*,
    int, int, int, cudaStream_t);
extern void launch_ssm_scan_fp32(const float*, const float*, float*,
    int, int, int, int, cudaStream_t);
extern void launch_rope_fp32(float*, const float*, const float*, int, int, int, int, cudaStream_t);
extern void build_qkv_fp32_kernel(const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);
extern void launch_flash_attn_fp32(const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);
extern void launch_transpose_attn(float*, const float*, int, int, int, cudaStream_t);

class MLALayer : public LayerFP32 {
    int d, H, Hkv, dh, dhr, dq, dc;
    const float *w_dqkv, *q_norm_w, *w_uq, *w_uk, *w_k2v, *w_qr, *w_kr, *c_proj_w;
    const float *rqc, *rqs, *rc, *rs;
public:
    bool init(const ModelConfig& cfg, const TensorMap& weights, int l) override {
        auto tn = [l](const char* c, const char* p) -> std::string {
            char b[128]; snprintf(b,128,"transformer.h.%d.%s.%s",l,c,p); return b;
        };
        auto ld = [&](const std::string& name, auto& ptr) {
            auto* t = weights.get(name); if (!t) return false;
            ptr = (const float*)t->data; return true;
        };
        d = cfg.dim; H = cfg.n_heads; Hkv = cfg.n_kv_heads;
        dh = cfg.head_dim; dhr = cfg.d_h_r ? cfg.d_h_r : 32; dq = dh + dhr;
        dc = cfg.d_c ? cfg.d_c : 160;
        if (!ld(tn("path","w_dqkv.weight"), w_dqkv)) return false;
        ld(tn("path","q_norm.weight"), q_norm_w);
        ld(tn("path","w_uq.weight"), w_uq);
        ld(tn("path","w_uk.weight"), w_uk);
        ld(tn("path","w_k2v.weight"), w_k2v);
        ld(tn("path","w_qr.weight"), w_qr);
        ld(tn("path","w_kr.weight"), w_kr);
        ld(tn("path","c_proj.weight"), c_proj_w);
        ld(tn("path","rope_q.cos"), rqc);
        ld(tn("path","rope_q.sin"), rqs);
        ld(tn("path","rope.cos"), rc);
        ld(tn("path","rope.sin"), rs);
        return true;
    }

    void forward(const float* path_in, const float* residual, float* ws,
                 int B, int T, cudaStream_t stream) override {
        int n = B * T;
        float* m = ws;
        float* ln1_save = m + n*dc + n*H*dh + n*Hkv*dh + 4*n*H*dq + 2*n*H*dh;

        launch_linear_fp32(path_in, w_dqkv, m, n, dc, d, stream);
        launch_pytorch_ln_kernel(m, q_norm_w, n, dc, 1e-5f, stream);

        cudaMemcpyAsync(ln1_save, path_in, n*d*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_linear_fp32(m, w_uq, const_cast<float*>(path_in), n, H*dh, dc, stream);

        int ok = n*H*dh, ov = ok+n*Hkv*dh, oq = ov+n*Hkv*dh, okr = oq+n*H*dhr;
        launch_linear_fp32(m, w_uk, m+ok, n, Hkv*dh, dc, stream);
        launch_linear_fp32(m+ok, w_k2v, m+ov, n, Hkv*dh, Hkv*dh, stream);
        launch_linear_fp32(ln1_save, w_qr, m+oq, n, H*dhr, d, stream);
        launch_linear_fp32(ln1_save, w_kr, m+okr, n, Hkv*dhr, d, stream);
        if (rqc) launch_rope_fp32(m+oq, rqc, rqs, B, T, H, dhr, stream);
        if (rc)  launch_rope_fp32(m+okr, rc, rs, B, T, Hkv, dhr, stream);

        float* attn_buf = m + okr + n*Hkv*dhr;
        float* Qf = attn_buf, *Kf = Qf+n*H*dq, *Vf = Kf+n*H*dq;
        build_qkv_fp32_kernel(path_in, m+ok, m+ov, m+oq, m+okr, Qf, Kf, Vf,
                              B, T, H, Hkv, dh, dhr, dq, stream);
        launch_flash_attn_fp32(Qf, Kf, Vf, Qf, B, H, T, dq, dh, stream);
        launch_transpose_attn(const_cast<float*>(path_in), Qf, H, T, dh, stream);
        launch_linear_fp32(path_in, c_proj_w, const_cast<float*>(path_in), n, d, H*dh, stream);
    }

    int workspace_per_token(int dim, int n_heads, int head_dim) const override {
        return dim * 4 * 2 / 3 / 256 * 256;
    }
};

static LayerFP32* create_mla() { return new MLALayer(); }

// ——— SSMLayer ———
class SSMLayer : public LayerFP32 {
    int d, H, dh, dc, ssm_steps, Hkv;
    const float *w_dq, *q_norm_w, *w_out;
    const float *w_mem[3], *w_decay[3];
public:
    bool init(const ModelConfig& cfg, const TensorMap& weights, int l) override {
        auto tn = [l](const char* c, const char* p) -> std::string {
            char b[128]; snprintf(b,128,"transformer.h.%d.%s.%s",l,c,p); return b;
        };
        auto ld = [&](const std::string& name, auto& ptr) {
            auto* t = weights.get(name); if (!t) return false;
            ptr = (const float*)t->data; return true;
        };
        d = cfg.dim; H = cfg.n_heads; dh = cfg.head_dim;
        Hkv = cfg.n_kv_heads; dc = cfg.d_c ? cfg.d_c : 160;
        ssm_steps = cfg.ssm_steps;
        if (!ld(tn("path","w_dq.weight"), w_dq)) return false;
        ld(tn("path","q_norm.weight"), q_norm_w);
        for (int k = 0; k < ssm_steps; k++) {
            char b[128];
            snprintf(b,128,tn("path","w_mem.%d.weight").c_str(),k);
            if (!ld(std::string(b), w_mem[k])) return false;
            snprintf(b,128,tn("path","w_decay.%d.weight").c_str(),k);
            if (!ld(std::string(b), w_decay[k])) return false;
        }
        if (!ld(tn("path","w_out.weight"), w_out)) return false;
        return true;
    }

    void forward(const float* path_in, const float* residual, float* ws,
                 int B, int T, cudaStream_t stream) override {
        int n = B * T;
        int ms = n * H * dh, ncq = n * dc;
        int doff = ncq + ssm_steps * ms;
        float* mems[3] = {ws+ncq, ws+ncq+ms, ws+ncq+2*ms};

        // cq = q_norm(w_dq(path_in))
        launch_linear_fp32(path_in, w_dq, ws, n, dc, d, stream);
        launch_pytorch_ln_kernel(ws, q_norm_w, n, dc, 1e-5f, stream);

        // w_mem.k + w_decay.k + sigmoid
        for (int k = 0; k < ssm_steps; k++) {
            launch_linear_fp32(ws, w_mem[k], mems[k], n, H*dh, dc, stream);
            launch_linear_fp32(ws, w_decay[k], ws+doff+k*n*H, n, H, dc, stream);
            launch_sigmoid_fp32(ws+doff+k*n*H, n*H, stream);
        }

        // ssm_agg + ssm_scan
        float* da = ws + doff;
        float* ma = da + n * H;
        launch_ssm_agg_fp32(mems[0], mems[1], mems[2],
            da, da+n*H, da+2*n*H, da, ma, H, dh, n, stream);
        launch_ssm_scan_fp32(ma, da, ma, B, T, H, dh, stream);

        // concat [ln1 || sfm] → w_out
        cudaMemcpyAsync(ws, path_in, n*d*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(ws+n*d, ma, n*H*dh*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_linear_fp32(ws, w_out, const_cast<float*>(path_in), n, d, d+H*dh, stream);
    }

    int workspace_per_token(int dim, int n_heads, int head_dim) const override {
        return dc + 3 * n_heads * head_dim; // ma + decays + cq
    }
};

static LayerFP32* create_ssm() { return new SSMLayer(); }

void register_layers_fp32() {
    register_layer_fp32("sparse_gather_fa", create_mla);
    register_layer_fp32("inertia_wave_ssm", create_ssm);
}
