// model_train_v2.cu — v2 training path
#include "model.h"
#include "model_v2_ctx.h"
#include "training/train.h"
#include "kernels/gemm.cuh"
#include <cstdio>
#include <cmath>

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_linear_dispatch(const void*,QuantType,const float*,float*,int,int,int,cudaStream_t);
extern float launch_crossentropy_fp32(const float*,const int*,float*,int,int,cudaStream_t);
extern void launch_adamw_fp32(float*,const float*,float*,float*,int,float,float,float,float,float,float,float,cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*,const float*,const float*,float*,float*,int,int,cudaStream_t);

float model_train_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream) {
    int n = B * T, d = cfg.dim, V = cfg.vocab_size;
    auto* w1_0 = w.get("transformer.h.0.mlp.w1.weight");
    int hd = (w1_0 && w1_0->n_dim >= 2) ? w1_0->shape[0] : (d * 4 * 2 / 3 / 256 * 256);

    if (!g_ctx.init(cfg, w)) return -1;
    int ws = compute_workspace_per_token(cfg, g_ctx.layers);
    g_ctx.ensure_bufs(n, d, ws, hd, V, true);

    auto& bufs = g_ctx.bufs;
    float* h = bufs.fwd.h;
    float* dh_ = bufs.grad.dh;
    float* da_ = bufs.grad.da;
    float* dm_ = bufs.grad.dm;
    float* dlm_ = bufs.grad.dlm;
    float* wg_ = bufs.wgrad.wg;
    float* base_save = bufs.fwd.save;

    int n_layers = (int)g_ctx.layers.size();

    // ═══════ FORWARD ═══════
    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte, ids, h, B, T, d, stream);

    for (int l = 0; l < n_layers; l++) {
        bufs.fwd.save = base_save + g_ctx.layers[l]->save_offset * n;
        g_ctx.layers[l]->forward(h, bufs.fwd, B, T, stream);
    }
    bufs.fwd.save = base_save;

    // Save h for ln_f backward at the end of save buffer
    int total_saved = 0;
    for (int i = 0; i < n_layers; i++)
        total_saved += g_ctx.layers[i]->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    float* lnx = base_save + total_saved * n;
    cudaMemcpyAsync(lnx, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto* ln_f_w = w.get("transformer.ln_f.weight");
    if (ln_f_w) {
        if (cfg.name.find("llama")==0)
            launch_rms_norm_fp32(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);
        else
            launch_pytorch_ln_kernel(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);
    }

    auto* lm_t = w.get("lm_head.weight");
    if (lm_t) {
        launch_linear_dispatch(lm_t->data, lm_t->quant_type, h, bufs.fwd.lm, n, V, d, stream);
    } else {
        launch_linear_fp32(h, wte, bufs.fwd.lm, n, V, d, stream);
    }
    float loss_val = launch_crossentropy_fp32(bufs.fwd.lm, targets, dlm_, n, V, stream);

    // ═══════ BACKWARD ═══════
    cudaMemsetAsync(dh_, 0, n * std::max(d, hd) * sizeof(float), stream);
    cudaMemsetAsync(da_, 0, n * std::max(d, hd) * sizeof(float), stream);
    cudaMemsetAsync(dm_, 0, n * ws * sizeof(float), stream);
    cudaStreamSynchronize(stream);

    cublasHandle_t ch = get_cublas_handle();
    cublasSetStream(ch, stream);
    float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;

    // lm_head backward
    auto* lm_bw_t = w.get("lm_head.weight");
    const float* lm_bw = lm_bw_t ? (const float*)lm_bw_t->data : wte;
    cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N,
                d, n, V, &a1, lm_bw, d, dlm_, V, &b1, dh_, d);
    // ln_f backward
    if (ln_f_w) {
        launch_layernorm_bwd_fp32(dh_, lnx, (const float*)ln_f_w->data, dh_, 0, n, d, stream);
    }

    // Layers backward (reverse order)
    for (int l = n_layers - 1; l >= 0; l--) {
        bufs.fwd.save = base_save + g_ctx.layers[l]->save_offset * n;
        g_ctx.layers[l]->backward(g_ctx.bufs.grad, bufs.fwd, wg_, B, T, stream);
    }
    bufs.fwd.save = base_save;

    // ═══════ OPTIMIZER ═══════
    {
        float lr = 3e-4f, beta1 = 0.9f, beta2 = 0.95f, eps_ = 1e-8f, wd = 0.1f;
        float b1p = powf(beta1, step + 1), b2p = powf(beta2, step + 1);
        size_t off = 0;
        for (auto& [n_, wt] : w.tensors) {
            if (wt.quant_type != QuantType::FP32) continue;
            launch_adamw_fp32((float*)wt.data, wg_ + off, bufs.wgrad.opt_m + off,
                             bufs.wgrad.opt_v + off, wt.n_elems,
                             lr, beta1, beta2, b1p, b2p, eps_, wd, stream);
            off += wt.n_elems;
        }
    }

    if (loss_d) cudaMemcpyAsync(loss_d, &loss_val, sizeof(float), cudaMemcpyHostToDevice, stream);
    return loss_val;
}
