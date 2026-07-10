#ifndef GGUF_RINA_BRIDGE_H
#define GGUF_RINA_BRIDGE_H

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <string>
#include <vector>
#include <map>
#include <cstdint>

struct BridgeConfig {
    int n_layers = 0;
    int dim = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    float rope_freq_base = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rms_norm_eps = 1e-5f;
};

struct BridgeModel {
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context * weight_ctx = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    BridgeConfig config;

    ~BridgeModel();
    BridgeModel() = default;
    BridgeModel(BridgeModel && other) noexcept
        : backend(other.backend), cpu_backend(other.cpu_backend), buft(other.buft), weight_buffer(other.weight_buffer),
          weight_ctx(other.weight_ctx), tensors(std::move(other.tensors)), config(other.config) {
        other.backend = nullptr;
        other.cpu_backend = nullptr;
        other.buft = nullptr;
        other.weight_buffer = nullptr;
        other.weight_ctx = nullptr;
    }
    BridgeModel(const BridgeModel &) = delete;
    BridgeModel & operator=(BridgeModel &&) = delete;
    BridgeModel & operator=(const BridgeModel &) = delete;
    bool valid() const { return backend != nullptr; }
};

bool bridge_load_model(const char * path, BridgeModel & model);

float * bridge_forward(BridgeModel & model, const int32_t * tokens, int n_tokens);

#endif // GGUF_RINA_BRIDGE_H
