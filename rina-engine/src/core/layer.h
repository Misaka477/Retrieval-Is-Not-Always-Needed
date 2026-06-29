#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

struct ModelConfig;
struct TensorMap;
struct ForwardBuffers;
struct GradBuffers;

// Forward declarations for per-arch vtable creation
struct LayerVTable;

// Unified Layer — type-erased wrapper around arch-specific implementations.
// All virtual dispatch goes through LayerVTable function pointers,
// avoiding vtable ABI mismatch between CUDA and C++ compilers.
class Layer {
public:
    void* impl = nullptr;
    const LayerVTable* vtab = nullptr;
    int save_offset = 0;

    Layer() = default;
    ~Layer();
    Layer(Layer&& other) noexcept;
    Layer& operator=(Layer&& other) noexcept;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    bool init(const ModelConfig& cfg, const TensorMap& weights, int layer_idx);
    void forward(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream);
    void backward(GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream);
    int workspace_per_token(int dim, int n_heads, int head_dim);
    int saved_per_token(int dim, int n_heads, int head_dim);

    // Factory — creates a fully constructed Layer from a config type string
    static std::unique_ptr<Layer> create(const std::string& type);
};

// Free function: create layer by type string, returns nullptr on unknown type
// Implemented in register_layers.cu (CUDA-compiled for vtable compatibility)
Layer* create_layer_by_type(const std::string& type);

// Factory registration (legacy)
using LayerFactory = Layer* (*)();
void register_layer_type(const std::string& type, LayerFactory f);

// Vtable structure — all functions take void* as first arg (the impl pointer)
struct LayerVTable {
    bool (*init_fn)(void* self, const ModelConfig& cfg, const TensorMap& weights, int layer_idx);
    void (*forward_fn)(void* self, float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream);
    void (*backward_fn)(void* self, GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream);
    int (*workspace_per_token_fn)(void* self, int dim, int n_heads, int head_dim);
    int (*saved_per_token_fn)(void* self, int dim, int n_heads, int head_dim);
    void (*destroy_fn)(void* self);
};
