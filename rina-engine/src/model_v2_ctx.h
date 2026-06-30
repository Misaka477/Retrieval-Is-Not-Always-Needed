#pragma once
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include <memory>
#include <vector>

// Cached model context shared by forward_v2 and train_v2
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

extern ModelContext g_ctx;
