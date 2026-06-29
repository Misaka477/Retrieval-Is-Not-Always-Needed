#include "core/layer.h"
#include "core/layer_fp32.h"
#include <unordered_map>

// ——— Layer RAII ———
Layer::~Layer() { if (vtab && vtab->destroy_fn) vtab->destroy_fn(impl); }
Layer::Layer(Layer&& other) noexcept : impl(other.impl), vtab(other.vtab), save_offset(other.save_offset) {
    other.impl = nullptr; other.vtab = nullptr;
}
Layer& Layer::operator=(Layer&& other) noexcept {
    if (this != &other) {
        if (vtab && vtab->destroy_fn) vtab->destroy_fn(impl);
        impl = other.impl; vtab = other.vtab; save_offset = other.save_offset;
        other.impl = nullptr; other.vtab = nullptr;
    }
    return *this;
}

bool Layer::init(const ModelConfig& cfg, const TensorMap& weights, int layer_idx) {
    return vtab->init_fn(impl, cfg, weights, layer_idx);
}
void Layer::forward(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream) {
    vtab->forward_fn(impl, h, bufs, B, T, stream);
}
void Layer::backward(GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream) {
    vtab->backward_fn(impl, grad, bufs, wg, B, T, stream);
}
int Layer::workspace_per_token(int dim, int n_heads, int head_dim) {
    return vtab->workspace_per_token_fn(impl, dim, n_heads, head_dim);
}
int Layer::saved_per_token(int dim, int n_heads, int head_dim) {
    return vtab->saved_per_token_fn(impl, dim, n_heads, head_dim);
}

// ——— Factory ———
std::unique_ptr<Layer> Layer::create(const std::string& type) {
    return std::unique_ptr<Layer>(create_layer_by_type(type));
}

// ——— fp16 Layer registry (legacy) ———
static std::unordered_map<std::string, LayerFactory>& reg16() {
    static std::unordered_map<std::string, LayerFactory> r;
    return r;
}
void register_layer(const std::string& type, LayerFactory f) { reg16()[type] = f; }
Layer* create_layer(const std::string& type) {
    auto it = reg16().find(type);
    return it != reg16().end() ? it->second() : nullptr;
}
void init_builtin_layers() {}

// ——— fp32 Layer registry ———
static std::unordered_map<std::string, LayerFP32Factory>& reg32() {
    static std::unordered_map<std::string, LayerFP32Factory> r;
    return r;
}
void register_layer_fp32(const std::string& type, LayerFP32Factory f) { reg32()[type] = f; }
LayerFP32* create_layer_fp32(const std::string& type) {
    auto it = reg32().find(type);
    return it != reg32().end() ? it->second() : nullptr;
}

void init_builtin_layers_fp32() {}

// ——— Old-style registry (keep for backward compat) ———
using OldLayerFactory = Layer* (*)();
static std::unordered_map<std::string, OldLayerFactory>& reg_new() {
    static std::unordered_map<std::string, OldLayerFactory> r;
    return r;
}
void register_layer_type(const std::string& type, OldLayerFactory f) { reg_new()[type] = f; }

extern void register_layers_fp32();
