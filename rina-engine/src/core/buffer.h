#pragma once
#include <cuda_runtime.h>
#include <cstdint>

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

    // KV cache for incremental inference
    // Per-layer: K[0..max_seq, kv_dim] then V[0..max_seq, kv_dim]
    // Layer i: K at offset (i*2) * max_seq * kv_dim, V at (i*2+1) * max_seq * kv_dim
    struct {
        float* data;          // [n_layers, 2, max_seq, kv_dim]
        int    max_seq;
        int    kv_dim;
        int    n_layers;
        int    start_pos;     // current sequence position (set before forward call)

        float* k(int layer) const {
            return data + (size_t)layer * 2 * max_seq * kv_dim;
        }
        float* v(int layer) const {
            return data + ((size_t)layer * 2 + 1) * max_seq * kv_dim;
        }
    } kv_cache;

    // Quantized KV cache for incremental inference (q2k_q1v)
    // K: q2_1 format (2-bit, 32 per block, 10 bytes/block)
    // V: q1_0 format (1-bit, 32 per block, 6 bytes/block)
    // Quantized KV cache
    // mode 0 = disabled (fp32), 1 = q2k_q1v, 2 = q4k (Q4_0 for both)
    // Storage: [layer0_K_blocks][layer0_V_blocks][layer1_K_blocks]...
    // K block size: mode 1→10 bytes, mode 2→18 bytes
    // V block size: mode 1→6 bytes,  mode 2→18 bytes
    struct {
        uint8_t* data;
        int    max_seq;
        int    kv_dim;
        int    n_layers;
        int    start_pos;
        int    mode;           // 0=off, 1=q8, 2=q4, 3=q4k_q2v, 4=q2, 5=q2k_q1v
        bool   pre_rope;       // true: cache K_raw (before RoPE)
        size_t blocks_per_layer;
        int    k_block_bytes;  // bytes per K block
        int    v_block_bytes;  // bytes per V block

        uint8_t* k(int layer) const {
            size_t off = (size_t)layer * blocks_per_layer * (k_block_bytes + v_block_bytes);
            return data + off;
        }
        uint8_t* v(int layer) const {
            size_t off = (size_t)layer * blocks_per_layer * (k_block_bytes + v_block_bytes)
                       + blocks_per_layer * k_block_bytes;
            return data + off;
        }
    } kv_cache_quant;

    // Scratch buffer for attention with KV cache expansion (optional)
    // Sized for Kf_expanded + Vf_expanded: [2, B, H, max_seq, head_dim]
    float* attn_scratch;
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
    void alloc_kv_cache(int n_layers, int max_seq, int n_kv_heads, int head_dim);
    void alloc_kv_cache_quant(int n_layers, int max_seq, int n_kv_heads, int head_dim, int mode);
    void alloc_attn_scratch(int B, int max_seq, int n_heads, int head_dim);

    void zero_grad(cudaStream_t stream);
    void free_all();
};
