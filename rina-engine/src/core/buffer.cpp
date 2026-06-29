#include "core/buffer.h"
#include <cstdio>
#include <algorithm>

static float* safe_malloc(size_t bytes) {
    float* p = nullptr;
    if (bytes > 0) cudaMalloc(&p, bytes);
    return p;
}

BufferManager::BufferManager()
    : n_cap(0), ws(0), hd(0), V(0), saved_per_layer(0) {
    fwd = ForwardBuffers{};
    grad = GradBuffers{};
    wgrad = WeightGradBuffers{};
}

BufferManager::~BufferManager() { free_all(); }

void BufferManager::alloc(int n, int d_model, int workspace_per_token,
                          int mlp_hd, int vocab_size, int n_layers,
                          int saved_per_layer_) {
    alloc_fwd(n, d_model, workspace_per_token, mlp_hd, vocab_size, saved_per_layer_);
    // Allocate fm/fl for training (needed by FlashAttention save_stats variant)
    if (!fwd.fm) fwd.fm  = safe_malloc(n * sizeof(float));
    if (!fwd.fl) fwd.fl  = safe_malloc(n * sizeof(float));
    alloc_grad(n, d_model, workspace_per_token, mlp_hd, vocab_size);
}

void BufferManager::alloc_fwd(int n, int d_model, int workspace_per_token,
                              int mlp_hd, int vocab_size, int saved_per_layer_) {
    if (n <= n_cap && ws >= workspace_per_token && hd >= mlp_hd && V >= vocab_size
        && saved_per_layer_ > 0 && saved_per_layer >= saved_per_layer_) return;
    free_all();
    n_cap = n;
    ws = workspace_per_token;
    hd = mlp_hd;
    V = vocab_size;
    saved_per_layer = saved_per_layer_;
    int fw = n * std::max(d_model, mlp_hd);
    fwd.h   = safe_malloc(fw * sizeof(float));
    fwd.a   = safe_malloc(fw * sizeof(float));
    fwd.m   = safe_malloc(n * ws * sizeof(float));
    fwd.lm  = safe_malloc(n * V * sizeof(float));
    // fm/fl only allocated when needed (training sets them via alloc() with save_stats flag)
    // For inference-only forward, they should stay null.
    fwd.save = safe_malloc(n * saved_per_layer * sizeof(float));
    if (!fwd.h || !fwd.a || !fwd.m || !fwd.lm || !fwd.save) {
        fprintf(stderr, "BufferManager: cudaMalloc failed (fwd)\n"); return;
    }
}

void BufferManager::alloc_grad(int n, int d_model, int workspace_per_token,
                               int mlp_hd, int vocab_size) {
    if (n <= n_cap && ws >= workspace_per_token && hd >= mlp_hd && V >= vocab_size
        && grad.dh) return;
    // Gradient buffers are allocated separately, never aliased with forward/weight
    int fw = n * std::max(d_model, mlp_hd);
    grad.dh  = safe_malloc(fw * sizeof(float));
    grad.da  = safe_malloc(fw * sizeof(float));
    grad.dm  = safe_malloc(n * ws * sizeof(float));
    grad.dlm = safe_malloc(n * V * sizeof(float));
    if (!grad.dh || !grad.da || !grad.dm || !grad.dlm) {
        fprintf(stderr, "BufferManager: cudaMalloc failed (grad)\n"); return;
    }
}

void BufferManager::alloc_wgrad(int total_weight_elems) {
    if (total_weight_elems <= wgrad.cap && wgrad.wg) return;
    cudaFree(wgrad.wg); cudaFree(wgrad.opt_m); cudaFree(wgrad.opt_v);
    wgrad.wg   = safe_malloc(total_weight_elems * sizeof(float));
    wgrad.opt_m = safe_malloc(total_weight_elems * sizeof(float));
    wgrad.opt_v = safe_malloc(total_weight_elems * sizeof(float));
    wgrad.cap = total_weight_elems;
    if (total_weight_elems > 0 && wgrad.wg) {
        cudaMemset(wgrad.wg, 0, total_weight_elems * sizeof(float));
        cudaMemset(wgrad.opt_m, 0, total_weight_elems * sizeof(float));
        cudaMemset(wgrad.opt_v, 0, total_weight_elems * sizeof(float));
    }
}

void BufferManager::zero_grad(cudaStream_t stream) {
    int fw = n_cap * std::max(hd, 1); // approximate, actual d_model/hd tracked internally
    if (grad.dh)  cudaMemsetAsync(grad.dh, 0, n_cap * hd * sizeof(float), stream);
    if (grad.da)  cudaMemsetAsync(grad.da, 0, n_cap * hd * sizeof(float), stream);
    if (grad.dm)  cudaMemsetAsync(grad.dm, 0, n_cap * ws * sizeof(float), stream);
    if (grad.dlm) cudaMemsetAsync(grad.dlm, 0, n_cap * V * sizeof(float), stream);
}

void BufferManager::free_all() {
    cudaFree(fwd.h);    fwd.h = nullptr;
    cudaFree(fwd.a);    fwd.a = nullptr;
    cudaFree(fwd.m);    fwd.m = nullptr;
    cudaFree(fwd.lm);   fwd.lm = nullptr;
    cudaFree(fwd.fm);   fwd.fm = nullptr;
    cudaFree(fwd.fl);   fwd.fl = nullptr;
    cudaFree(fwd.save); fwd.save = nullptr;
    cudaFree(grad.dh);  grad.dh = nullptr;
    cudaFree(grad.da);  grad.da = nullptr;
    cudaFree(grad.dm);  grad.dm = nullptr;
    cudaFree(grad.dlm); grad.dlm = nullptr;
    cudaFree(wgrad.wg);   wgrad.wg = nullptr;
    cudaFree(wgrad.opt_m); wgrad.opt_m = nullptr;
    cudaFree(wgrad.opt_v); wgrad.opt_v = nullptr;
    n_cap = 0; ws = 0; hd = 0; V = 0;
    wgrad.cap = 0;
}
