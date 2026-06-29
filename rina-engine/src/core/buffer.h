#pragma once
#include <cuda_runtime.h>

// Forward pass buffers — all pointers are device memory
struct ForwardBuffers {
    float *h;       // [B*T, d_model] residual stream
    float *a;       // [B*T, max(d_model, hd)] activation / MLP hidden
    float *m;       // [B*T, ws] workspace for per-arch intermediates
    float *lm;      // [B*T, vocab_size] logits
    float *save;    // [B*T, saved_per_layer * n_layers] saved intermediates for backward
    int cap;        // current B*T capacity

    float *fm;      // [B*T] flash-attention softmax max (per-query)
    float *fl;      // [B*T] flash-attention softmax lse (per-query)
};

// Gradient buffers — separate pool, never aliases weight or forward buffers
struct GradBuffers {
    float *dh;      // gradient of h
    float *da;      // gradient of a
    float *dm;      // gradient of m (workspace)
    float *dlm;     // gradient of lm (logits)
};

// Weight gradient + optimizer state — contiguous, indexed by weight offset
struct WeightGradBuffers {
    float *wg;      // weight gradients (one float per weight element)
    float *opt_m;   // AdamW momentum
    float *opt_v;   // AdamW variance
    int cap;        // number of float elements allocated
};

// Unified buffer manager — owns all device memory
struct BufferManager {
    ForwardBuffers fwd;
    GradBuffers grad;
    WeightGradBuffers wgrad;

    int n_cap;      // current B*T capacity (for fwd/grad)
    int ws;         // workspace size per token
    int hd;         // MLP hidden dim
    int V;          // vocab size
    int saved_per_layer;

    BufferManager();
    ~BufferManager();

    void alloc(int n, int d_model, int workspace_per_token,
               int mlp_hd, int vocab_size, int n_layers,
               int saved_per_layer_);

    void alloc_fwd(int n, int d_model, int workspace_per_token,
                   int mlp_hd, int vocab_size, int saved_per_layer_);
    void alloc_grad(int n, int d_model, int workspace_per_token,
                    int mlp_hd, int vocab_size);
    void alloc_wgrad(int total_weight_elems);

    void zero_grad(cudaStream_t stream);
    void free_all();
};
