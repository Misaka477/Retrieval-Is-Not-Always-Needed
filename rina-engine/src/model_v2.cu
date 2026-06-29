#include "model.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "kernels/gemm.cuh"
#include "training/train.h"
#include <cstdio>
#include <cmath>
#include <memory>
#include <vector>

// ── Kernel declarations ──
extern void launch_embedding_fp32(const float*, const int*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern float launch_crossentropy_fp32(const float*, const int*, float*, int, int, cudaStream_t);
extern void launch_adamw_fp32(float*, const float*, float*, float*, int,
    float, float, float, float, float, float, float, cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, int, cudaStream_t);

// ── Cached model context ──
struct ModelContext {
    std::vector<std::unique_ptr<Layer>> layers;
    BufferManager bufs;
    const ModelConfig* cfg = nullptr;
    const TensorMap* w = nullptr;

    bool init(const ModelConfig& cfg_, const TensorMap& w_) {
        if (cfg && cfg->n_layers == cfg_.n_layers && cfg->dim == cfg_.dim
            && cfg->n_heads == cfg_.n_heads && cfg->vocab_size == cfg_.vocab_size
            && (int)layers.size() == cfg_.n_layers)
            return true;
        cfg = &cfg_; w = &w_;
        layers = build_layers(cfg_, w_);
        return !layers.empty();
    }

    void ensure_bufs(int n, int d, int ws, int hd, int V, bool training) {
        int n_layers = (int)layers.size();
        int total_saved = 0;
        for (auto& l : layers)
            total_saved += l->saved_per_token(d, cfg->n_heads, cfg->head_dim);
        bufs.alloc_fwd(n, d, ws, hd, V, total_saved);
        if (training) {
            bufs.alloc_grad(n, d, ws, hd, V);
            if (!bufs.fwd.fm) cudaMalloc(&bufs.fwd.fm, n * sizeof(float));
            if (!bufs.fwd.fl) cudaMalloc(&bufs.fwd.fl, n * sizeof(float));
            if (!bufs.wgrad.wg) {
                size_t total_w = 0;
                for (auto& [n_, wt] : w->tensors)
                    if (wt.quant_type == QuantType::FP32) total_w += wt.n_elems;
                if (total_w > 0) bufs.alloc_wgrad((int)total_w);
            }
        }
    }
};

static ModelContext g_ctx;

static float* layer_save_ptr(int layer_idx, int n, int d, int n_heads, int head_dim) {
    int offset = 0;
    for (int i = 0; i < layer_idx; i++)
        offset += g_ctx.layers[i]->saved_per_token(d, n_heads, head_dim);
    return g_ctx.bufs.fwd.save + offset * n;
}

// ── model_forward_v2 ──
void model_forward_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream) {
    int n = B * T, d = cfg.dim, V = cfg.vocab_size;
    int hd = d * 4 * 2 / 3 / 256 * 256;

    if (!g_ctx.init(cfg, w)) return;
    int ws = compute_workspace_per_token(cfg, g_ctx.layers);
    g_ctx.ensure_bufs(n, d, ws, hd, V, false);

    auto& bufs = g_ctx.bufs;
    float* h = bufs.fwd.h;
    float* base_save = bufs.fwd.save;

    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    if (!wte) { fprintf(stderr, "v2: wte null\n"); return; }
    launch_embedding_fp32(wte, ids, h, B, T, d, stream);

    for (int l = 0; l < (int)g_ctx.layers.size(); l++) {
        bufs.fwd.save = base_save + g_ctx.layers[l]->save_offset * n;
        g_ctx.layers[l]->forward(h, bufs.fwd, B, T, stream);
    }
    bufs.fwd.save = base_save;

    auto* ln_f_w = w.get("transformer.ln_f.weight");
    if (ln_f_w) launch_pytorch_ln_kernel(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);

    const float* lm_w = nullptr;
    auto* lm_t = w.get("lm_head.weight");
    if (lm_t) lm_w = (const float*)lm_t->data;
    if (!lm_w) lm_w = wte;
    launch_linear_fp32(h, lm_w, bufs.fwd.lm, n, V, d, stream);

    cudaMemcpyAsync(logits, bufs.fwd.lm, n * V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

// ── model_train_v2 ──
float model_train_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream) {
    int n = B * T, d = cfg.dim, V = cfg.vocab_size;
    int hd = d * 4 * 2 / 3 / 256 * 256;

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
    // Use a spare area in the save buffer (past all layers' save regions)
    cudaMemcpyAsync(lnx, h, n * d * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto* ln_f_w = w.get("transformer.ln_f.weight");
    if (ln_f_w) launch_pytorch_ln_kernel(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);

    const float* lm_w = nullptr;
    auto* lm_t = w.get("lm_head.weight");
    if (lm_t) lm_w = (const float*)lm_t->data;
    if (!lm_w) lm_w = wte;
    launch_linear_fp32(h, lm_w, bufs.fwd.lm, n, V, d, stream);
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
    cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N,
                d, n, V, &a1, lm_w, d, dlm_, V, &b1, dh_, d);
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
    {   // Use a block for local vars
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
