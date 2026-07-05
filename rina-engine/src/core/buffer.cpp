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
    fwd.save = safe_malloc(n * saved_per_layer * sizeof(float));
    fwd.attn_scratch = nullptr;
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

void BufferManager::alloc_kv_cache(int n_layers, int max_seq, int n_kv_heads, int head_dim) {
    int kv_dim = n_kv_heads * head_dim;
    size_t bytes = (size_t)n_layers * 2 * max_seq * kv_dim * sizeof(float);
    cudaFree(fwd.kv_cache.data);
    fwd.kv_cache.data = nullptr;
    cudaMalloc(&fwd.kv_cache.data, bytes);
    if (fwd.kv_cache.data) cudaMemset(fwd.kv_cache.data, 0, bytes);
    fwd.kv_cache.max_seq = max_seq;
    fwd.kv_cache.kv_dim = kv_dim;
    fwd.kv_cache.n_layers = n_layers;
    fwd.kv_cache.start_pos = 0;
}

// Mode → block sizes (in bytes per 32-element block):
// 0=fp32(no quant) 1=q8:34+34  2=q4:18+18  3=q4k_q2v:18+10  4=q2:10+10  5=q2k_q1v:10+6
static void mode_block_sizes(int mode, int& kbb, int& vbb) {
    switch (mode) {
        case 1: kbb = 34; vbb = 34; break; // q8
        case 2: kbb = 18; vbb = 18; break; // q4
        case 3: kbb = 18; vbb = 10; break; // q4k_q2v
        case 4: kbb = 10; vbb = 10; break; // q2
        case 5: kbb = 10; vbb = 6;  break; // q2k_q1v
        default: kbb = 4; vbb = 4; break;  // fp32 (fallback — not used)
    }
}

void BufferManager::alloc_kv_cache_quant(int n_layers, int max_seq, int n_kv_heads, int head_dim, int mode) {
    int kv_dim = n_kv_heads * head_dim;
    size_t blocks_per_layer = (size_t)max_seq * kv_dim / 32;
    fwd.kv_cache_quant.mode = mode;
    fwd.kv_cache_quant.pre_rope = false;
    mode_block_sizes(mode, fwd.kv_cache_quant.k_block_bytes, fwd.kv_cache_quant.v_block_bytes);
    fwd.kv_cache_quant.blocks_per_layer = blocks_per_layer;

    size_t per_layer = blocks_per_layer * (fwd.kv_cache_quant.k_block_bytes + fwd.kv_cache_quant.v_block_bytes);
    cudaFree(fwd.kv_cache_quant.data);
    fwd.kv_cache_quant.data = nullptr;
    cudaMalloc(&fwd.kv_cache_quant.data, per_layer * n_layers);
    if (fwd.kv_cache_quant.data) cudaMemset(fwd.kv_cache_quant.data, 0, per_layer * n_layers);
    fwd.kv_cache_quant.max_seq = max_seq;
    fwd.kv_cache_quant.kv_dim = kv_dim;
    fwd.kv_cache_quant.n_layers = n_layers;
    fwd.kv_cache_quant.start_pos = 0;
}

void BufferManager::alloc_attn_scratch(int B, int max_seq, int n_heads, int head_dim) {
    size_t bytes = (size_t)B * max_seq * n_heads * head_dim * 3 * sizeof(float); // Qf + Kf + Vf
    cudaFree(fwd.attn_scratch);
    fwd.attn_scratch = nullptr;
    cudaMalloc(&fwd.attn_scratch, bytes);
    if (fwd.attn_scratch) cudaMemset(fwd.attn_scratch, 0, bytes);
}

void BufferManager::free_all() {
    cudaFree(fwd.h);    fwd.h = nullptr;
    cudaFree(fwd.a);    fwd.a = nullptr;
    cudaFree(fwd.m);    fwd.m = nullptr;
    cudaFree(fwd.lm);   fwd.lm = nullptr;
    cudaFree(fwd.fm);   fwd.fm = nullptr;
    cudaFree(fwd.fl);   fwd.fl = nullptr;
    cudaFree(fwd.save); fwd.save = nullptr;
    cudaFree(fwd.kv_cache_quant.data); fwd.kv_cache_quant.data = nullptr;
    cudaFree(fwd.kv_cache_quant.data); fwd.kv_cache_quant.data = nullptr;
    cudaFree(fwd.kv_cache.data);   fwd.kv_cache.data = nullptr;
    cudaFree(fwd.attn_scratch);    fwd.attn_scratch = nullptr;
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
