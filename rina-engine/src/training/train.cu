#include "training/train.h"
#include "core/config.h"
#include "core/tensor.h"

std::vector<std::unique_ptr<Layer>> build_layers(const ModelConfig& cfg, const TensorMap& weights) {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.reserve(cfg.n_layers);

    int accum_offset = 0;
    for (int l = 0; l < cfg.n_layers; l++) {
        std::string type;
        if (l < (int)cfg.layers.size()) {
            type = cfg.layers[l].type;
        } else {
            type = "inertia_wave_ssm";
        }

        auto layer = std::unique_ptr<Layer>(create_layer_by_type(type));
        if (!layer) {
            fprintf(stderr, "build_layers: unknown layer type '%s' at index %d\n", type.c_str(), l);
            return {};
        }
        if (!layer->init(cfg, weights, l)) {
            fprintf(stderr, "build_layers: init failed for layer %d type '%s'\n", l, type.c_str());
            return {};
        }
        layer->save_offset = accum_offset;
        accum_offset += layer->saved_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
        layers.push_back(std::move(layer));
    }
    return layers;
}

int compute_workspace_per_token(const ModelConfig& cfg,
                                 const std::vector<std::unique_ptr<Layer>>& layers) {
    int max_ws = 0;
    for (auto& layer : layers) {
        int ws = layer->workspace_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
        if (ws > max_ws) max_ws = ws;
    }
    return max_ws;
}

int compute_saved_per_token(const ModelConfig& cfg,
                             const std::vector<std::unique_ptr<Layer>>& layers) {
    int total = 0;
    for (auto& layer : layers) {
        total += layer->saved_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
    }
    return total;
}
