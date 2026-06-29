#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/config.h"
#include "core/tensor.h"
#include "core/sparse_index.h"

// Custom kernel declarations
extern void launch_embedding_fp32(const float*, const int*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_flash_attn_fp32(const float*, const float*, const float*, float*,
    int, int, int, int, int, cudaStream_t);
extern void launch_silu_fp32(float*, int, cudaStream_t);

// How to use when `--pytorch` is passed:
// Engine loads weights via loader.cpp as usual.
// pt_init() copies them into torch Tensors on GPU.
// pt_forward() runs the whole forward using ONLY PiTorch ops.
// Compare output tokens with model_forward_fp32().

struct PTModel {
    torch::Device dev;
    int n_layers, ssm_steps, d, H, Hkv, dh, dhr, dq, dc, hd, vocab_size;
    std::vector<int> layer_types;  // 0=SSM, 1=Attention
    const TensorMap* wm;
    bool custom_embed, custom_ln, custom_cproj, custom_mlp, custom_attn;
    SparseIndexManager sim{32, 8, 4};

    torch::Tensor wte, ln_f_w;
    struct LayerW {
        torch::Tensor ln1_w, ln2_w, q_norm_w;
        // MLA (attention layers)
        torch::Tensor w_dqkv, w_uq, w_uk, w_k2v, w_qr, w_kr, c_proj;
        // SSM layers
        torch::Tensor w_dq, w_mem[3], w_decay[3], w_out;
        // MLP (shared)
        torch::Tensor w1, w2, w3;
        // RoPE (attention layers only)
        torch::Tensor rqc, rqs, rc, rs;
    };
    std::vector<LayerW> layers;

    auto _tn(int l, const char* c, const char* p) -> std::string {
        char b[128]; snprintf(b,128,"transformer.h.%d.%s.%s",l,c,p); return b;
    }
    torch::Tensor _load(const TensorMap& wm, const std::string& name, std::vector<int64_t> shape) {
        auto* t = wm.get(name);
        if (!t || !t->data) { fprintf(stderr,"MISSING: %s\n",name.c_str()); return torch::empty(shape); }
        return torch::from_blob((void*)t->data, torch::IntArrayRef(shape),
            torch::TensorOptions().dtype(torch::kFloat32).device(dev)).clone();
    }
    torch::Tensor _load1d(const TensorMap& wm, const std::string& name, int d0) {
        return _load(wm, name, {d0});
    }

    PTModel(const ModelConfig& cfg, const TensorMap& wm,
            bool custom_embed=false, bool custom_ln=false, bool custom_attn=false,
            bool custom_cproj=false, bool custom_mlp=false)
        : dev(torch::kCUDA, 0), custom_embed(custom_embed), custom_ln(custom_ln),
          custom_attn(custom_attn), custom_cproj(custom_cproj), custom_mlp(custom_mlp),
          wm(&wm) {
        n_layers = cfg.n_layers; d = cfg.dim;
        H = cfg.n_heads; Hkv = cfg.n_kv_heads; dh = cfg.head_dim;
        dhr = cfg.d_h_r ? cfg.d_h_r : 32; dq = dh + dhr;
        dc = cfg.d_c ? cfg.d_c : 160; ssm_steps = cfg.ssm_steps;
        hd = d * 4 * 2 / 3 / 256 * 256;
        vocab_size = cfg.vocab_size;

        // Store layer types
        layer_types.reserve(n_layers);
        for (int l = 0; l < n_layers; l++)
            layer_types.push_back(l < (int)cfg.layers.size() ? cfg.layers[l].layer_type : 0);

        wte = _load(wm, "transformer.wte.weight", {vocab_size, d});
        ln_f_w = _load1d(wm, "transformer.ln_f.weight", d);

        layers.resize(n_layers);
        for (int l = 0; l < n_layers; l++) {
            auto& L = layers[l];
            int lt = layer_types[l];
            L.ln1_w    = _load1d(wm, _tn(l,"ln1","weight"), d);
            L.ln2_w    = _load1d(wm, _tn(l,"ln2","weight"), d);
            L.q_norm_w = _load1d(wm, _tn(l,"path","q_norm.weight"), dc);
            if (lt == 0) {
                // SSM weights
                L.w_dq    = _load(wm, _tn(l,"path","w_dq.weight"), {dc, d});
                for (int k = 0; k < ssm_steps; k++) {
                    char b[128];
                    snprintf(b,128,_tn(l,"path","w_mem.%d.weight").c_str(),k);
                    L.w_mem[k] = _load(wm, std::string(b), {H*dh, dc});
                    snprintf(b,128,_tn(l,"path","w_decay.%d.weight").c_str(),k);
                    L.w_decay[k] = _load(wm, std::string(b), {H, dc});
                }
                L.w_out   = _load(wm, _tn(l,"path","w_out.weight"), {d, d+H*dh});
            } else {
                // MLA weights
                L.w_dqkv  = _load(wm, _tn(l,"path","w_dqkv.weight"), {dc, d});
                L.w_uq    = _load(wm, _tn(l,"path","w_uq.weight"), {H*dh, dc});
                L.w_uk    = _load(wm, _tn(l,"path","w_uk.weight"), {Hkv*dh, dc});
                L.w_k2v   = _load(wm, _tn(l,"path","w_k2v.weight"), {Hkv*dh, Hkv*dh});
                L.w_qr    = _load(wm, _tn(l,"path","w_qr.weight"), {H*dhr, d});
                L.w_kr    = _load(wm, _tn(l,"path","w_kr.weight"), {Hkv*dhr, d});
                L.c_proj  = _load(wm, _tn(l,"path","c_proj.weight"), {d, H*dh});
                L.rqc = _load(wm, _tn(l,"path","rope_q.cos"), {512, dhr/2});
                L.rqs = _load(wm, _tn(l,"path","rope_q.sin"), {512, dhr/2});
                L.rc  = _load(wm, _tn(l,"path","rope.cos"),   {512, dhr/2});
                L.rs  = _load(wm, _tn(l,"path","rope.sin"),   {512, dhr/2});
            }
            // MLP weights (shared)
            L.w1      = _load(wm, _tn(l,"mlp","w1.weight"), {hd, d});
            L.w2      = _load(wm, _tn(l,"mlp","w2.weight"), {d, hd});
            L.w3      = _load(wm, _tn(l,"mlp","w3.weight"), {hd, d});
        }
    }

    void forward(const int* ids_d, float* logits_d, int B, int T, cudaStream_t stream) {
        auto dtype = torch::kFloat32;
        auto int_opts = torch::TensorOptions().dtype(torch::kInt64).device(dev);
        auto ids_i64 = torch::from_blob((void*)ids_d, {B, T}, torch::TensorOptions().dtype(torch::kInt32).device(dev)).to(torch::kInt64);
        auto ids = ids_i64;

        torch::Tensor x, h;
        // Embedding
        if (custom_embed) {
            auto wte_data = wm->get("transformer.wte.weight");
            if (wte_data && wte_data->data) {
                // Use a temporary torch tensor as the output buffer
                x = torch::empty({B, T, d}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                launch_embedding_fp32((const float*)wte_data->data, ids_d,
                                     x.data_ptr<float>(), B, T, d, stream);
            } else {
                x = torch::embedding(wte, ids).to(dtype);
            }
        } else {
            x = torch::embedding(wte, ids).to(dtype);
        }
        h = x;

        for (int l = 0; l < n_layers; l++) {
            auto& L = layers[l];
            // LN1
            auto ln1 = h.clone();
            if (custom_ln) {
                launch_pytorch_ln_kernel(ln1.data_ptr<float>(), L.ln1_w.data_ptr<float>(), B*T, d, 1e-5, stream);
            } else {
                { auto m=ln1.mean(-1,true); auto v=ln1.var(-1,false,true);
                  ln1 = (ln1-m)/(v+1e-5).sqrt(); ln1 = ln1 * L.ln1_w; }
            }
            if (layer_types[l] == 0) {
                // SSM forward
                auto cq = torch::matmul(ln1, L.w_dq.transpose(0,1));
                if (custom_ln) {
                    launch_pytorch_ln_kernel(cq.data_ptr<float>(), L.q_norm_w.data_ptr<float>(), B*T, dc, 1e-5, stream);
                } else {
                    { auto m=cq.mean(-1,true); auto v=cq.var(-1,false,true);
                      cq = (cq-m)/(v+1e-5).sqrt(); cq = cq * L.q_norm_w; }
                }
                auto mems = std::vector<torch::Tensor>(ssm_steps);
                auto decays = std::vector<torch::Tensor>(ssm_steps);
                for (int k = 0; k < ssm_steps; k++) {
                    mems[k] = torch::matmul(cq, L.w_mem[k].transpose(0,1)).view({B,T,H,dh});
                    decays[k] = torch::sigmoid(torch::matmul(cq, L.w_decay[k].transpose(0,1))).view({B,T,H,1});
                }
                auto d_agg = decays[0] * decays[1] * (ssm_steps > 2 ? decays[2] : torch::ones_like(decays[0]));
                auto m_agg = mems[0] * decays[1] * (ssm_steps > 2 ? decays[2] : torch::ones_like(decays[0]))
                           + mems[1] * (ssm_steps > 2 ? decays[2] : torch::ones_like(decays[0]))
                           + (ssm_steps > 2 ? mems[2] : torch::zeros_like(mems[0]));
                // SSM scan: cumsum in log space
                auto a = d_agg.expand({-1,-1,-1,dh});
                auto lca = torch::cumsum(a.log(), 1);
                auto ca = lca.exp();
                auto sf = ca * torch::cumsum(m_agg / (ca + 1e-8), 1);
                sf = sf.reshape({B,T,-1});
                // w_out(concat(ln1, sf))
                auto cat = torch::cat({ln1, sf}, -1);
                auto out = torch::matmul(cat, L.w_out.transpose(0,1));
                h = h + out;
            } else {
                // MLA forward
                auto cq = torch::matmul(ln1, L.w_dqkv.transpose(0,1));
                if (custom_ln) {
                    launch_pytorch_ln_kernel(cq.data_ptr<float>(), L.q_norm_w.data_ptr<float>(), B*T, dc, 1e-5, stream);
                } else {
                    { auto m=cq.mean(-1,true); auto v=cq.var(-1,false,true);
                      cq = (cq-m)/(v+1e-5).sqrt(); cq = cq * L.q_norm_w; }
                }
            auto qc = torch::matmul(cq, L.w_uq.transpose(0,1)).view({B,T,H,dh}).transpose(1,2);
            auto kc = torch::matmul(cq, L.w_uk.transpose(0,1)).view({B,T,Hkv,dh}).transpose(1,2);
            auto kf = torch::matmul(cq, L.w_uk.transpose(0,1)).reshape({B*T,Hkv*dh});
            auto v  = torch::matmul(kf, L.w_k2v.transpose(0,1)).view({B,T,Hkv,dh}).transpose(1,2);
            // qr, kr with RoPE
            auto qr = torch::matmul(ln1, L.w_qr.transpose(0,1)).view({B,T,H,dhr}).transpose(1,2);
            auto kr = torch::matmul(ln1, L.w_kr.transpose(0,1)).view({B,T,Hkv,dhr}).transpose(1,2);
            // RoPE
            {
                int hlf = dhr / 2;
                auto rope = [&](torch::Tensor& x, const torch::Tensor& cos_t, const torch::Tensor& sin_t, int H_dim) {
                    auto cd = cos_t.slice(0,0,T).reshape({1,1,T,hlf}); // [1,1,T,hlf]
                    auto sd = sin_t.slice(0,0,T).reshape({1,1,T,hlf}); // [1,1,T,hlf]
                    auto x0 = x.index({"...",torch::indexing::Slice(0,dhr,2)});
                    auto x1 = x.index({"...",torch::indexing::Slice(1,dhr,2)});
                    x.copy_(torch::cat({x0*cd - x1*sd, x0*sd + x1*cd}, -1).reshape(x.sizes()));
                };
                auto x0_q = qr.index({"...",torch::indexing::Slice(0,dhr,2)});
                auto x1_q = qr.index({"...",torch::indexing::Slice(1,dhr,2)});
                auto cd_q = L.rqc.slice(0,0,T).reshape({1,1,T,hlf});
                auto sd_q = L.rqs.slice(0,0,T).reshape({1,1,T,hlf});
                qr = torch::cat({x0_q*cd_q - x1_q*sd_q, x0_q*sd_q + x1_q*cd_q}, -1).view({B,H,T,dhr});
                auto x0_k = kr.index({"...",torch::indexing::Slice(0,dhr,2)});
                auto x1_k = kr.index({"...",torch::indexing::Slice(1,dhr,2)});
                auto cd_k = L.rc.slice(0,0,T).reshape({1,1,T,hlf});
                auto sd_k = L.rs.slice(0,0,T).reshape({1,1,T,hlf});
                kr = torch::cat({x0_k*cd_k - x1_k*sd_k, x0_k*sd_k + x1_k*cd_k}, -1).view({B,Hkv,T,dhr});
            }
            int rep = H / Hkv;
            if (rep > 1) {
                kc = kc.repeat_interleave(rep, 1);
                v  = v.repeat_interleave(rep, 1);
                kr = kr.repeat_interleave(rep, 1);
            }
            auto q = torch::cat({qc,qr}, -1);
            auto k = torch::cat({kc,kr}, -1);
            torch::Tensor attn_result;
            if (custom_attn) {
                // FlashAttention-style tiled attention
                auto q_bht = q.transpose(0,1).contiguous();  // [H, B, T, dq]
                auto k_bht = k.transpose(0,1).contiguous();
                auto v_bht = v.transpose(0,1).contiguous();
                int Bh = B * H;
                auto Q_flat = q_bht.reshape({Bh, -1, dq});  // [Bh, T, dq]
                auto K_flat = k_bht.reshape({Bh, -1, dq});
                auto V_flat = v_bht.reshape({Bh, -1, dh});
                attn_result = torch::empty({Bh, T, dh}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                launch_flash_attn_fp32(Q_flat.data_ptr<float>(), K_flat.data_ptr<float>(),
                    V_flat.data_ptr<float>(), attn_result.data_ptr<float>(),
                    B, H, T, dq, dh, stream);
                attn_result = attn_result.view({B*T, H*dh});
            } else {
                // Full attention via PyTorch SDPA
                attn_result = torch::scaled_dot_product_attention(q, k, v, {}, 0.0, true);
                attn_result = attn_result.transpose(1,2).contiguous().view({B*T, H*dh});
            }
            // c_proj
            torch::Tensor out;
            if (custom_cproj) {
                out = torch::empty({B*T, d}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                launch_linear_fp32(attn_result.data_ptr<float>(), L.c_proj.data_ptr<float>(),
                                   out.data_ptr<float>(), B*T, d, H*dh, stream);
                out = out.view({B,T,-1});
            } else {
                out = torch::matmul(attn_result, L.c_proj.transpose(0,1)).view({B,T,-1});
            }
            h = h + out;
            }
            // LN2 + MLP (shared by SSM and attention paths)
            auto r = h.clone();
            if (custom_ln) {
                launch_pytorch_ln_kernel(r.data_ptr<float>(), L.ln2_w.data_ptr<float>(), B*T, d, 1e-5, stream);
            } else {
                { auto m2=r.mean(-1,true); auto v2=r.var(-1,false,true);
                  r = (r-m2)/(v2+1e-5).sqrt(); r = r * L.ln2_w; }
            }
            if (custom_mlp) {
                // Custom matmuls + PyTorch silu
                auto gate = torch::empty({B*T, hd}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                auto up   = torch::empty({B*T, hd}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                launch_linear_fp32(r.data_ptr<float>(), L.w1.data_ptr<float>(),
                                   gate.data_ptr<float>(), B*T, hd, d, stream);
                launch_linear_fp32(r.data_ptr<float>(), L.w3.data_ptr<float>(),
                                   up.data_ptr<float>(), B*T, hd, d, stream);
                launch_silu_fp32(gate.data_ptr<float>(), B*T*hd, stream);
                auto prod = gate * up;
                auto mlp = torch::empty({B*T, d}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
                launch_linear_fp32(prod.data_ptr<float>(), L.w2.data_ptr<float>(),
                                   mlp.data_ptr<float>(), B*T, d, hd, stream);
                h = h + mlp;
            } else {
                auto gate = torch::silu(torch::matmul(r, L.w1.transpose(0,1)));
                auto up   = torch::matmul(r, L.w3.transpose(0,1));
                auto mlp  = torch::matmul(gate*up, L.w2.transpose(0,1));
                h = h + mlp;
            }
        }
        // ln_f
        auto xf = h.clone();
        if (custom_ln) {
            launch_pytorch_ln_kernel(xf.data_ptr<float>(), ln_f_w.data_ptr<float>(), B*T, d, 1e-5, stream);
        } else {
            { auto m3=xf.mean(-1,true); auto v3=xf.var(-1,false,true);
              xf = (xf-m3)/(v3+1e-5).sqrt(); xf = xf * ln_f_w; }
        }
        auto logits = torch::matmul(xf, wte.transpose(0,1));
        cudaMemcpyAsync(logits_d, logits.data_ptr<float>(), B*T*vocab_size*sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
};

struct PtFlags { bool embed=false,ln=false,attn=false,cproj=false,mlp=false; };

static PTModel* g_pt = nullptr;

extern "C" void pt_init(const ModelConfig& cfg, const TensorMap& w, bool custom_embed, bool custom_ln,
    bool custom_attn, bool custom_cproj, bool custom_mlp) {
    if (g_pt) delete g_pt;
    g_pt = new PTModel(cfg, w, custom_embed, custom_ln, custom_attn, custom_cproj, custom_mlp);
    fprintf(stderr,"PyTorch model initialized on CUDA.\n");
}
extern "C" void pt_forward(const int* ids, float* logits, int B, int T, cudaStream_t s) {
    if (g_pt) g_pt->forward(ids, logits, B, T, s);
    else fprintf(stderr,"ERROR: pt_init not called\n");
}
extern "C" void pt_free() { delete g_pt; g_pt = nullptr; }
