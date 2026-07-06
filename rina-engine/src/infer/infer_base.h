#pragma once
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

// ── Inference interface (per-architecture) ───────────────────
struct Inference {
    virtual ~Inference() = default;

    // Initialize: allocate all buffers, build layer pipeline
    virtual bool init(ModelConfig& cfg, const TensorMap& weights) = 0;

    // Forward: run full inference on one batch
    // ids: [B, T] input token IDs
    // logits: [B, T, V] output logits (device ptr)
    // start_pos: position in KV cache (0 for prefill, >0 for incremental decode)
    virtual void forward(const int* ids, float* logits,
                         int B, int T, int start_pos, cudaStream_t stream) = 0;
};

// ── Factory ──────────────────────────────────────────────────
using InferenceFactory = Inference* (*)();
void register_inference(const std::string& name, InferenceFactory factory);
Inference* create_inference(const std::string& name);

// ── Built-in inference registrations ─────────────────────────
// Called once at startup to register all architecture-specific implementations.
void register_all_inferences();
