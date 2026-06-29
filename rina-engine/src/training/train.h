#pragma once
#include "core/layer.h"
#include "core/buffer.h"
#include <memory>
#include <vector>

struct ModelConfig;
struct TensorMap;

// Build layers from config — creates Layer objects for all layers in the model
std::vector<std::unique_ptr<Layer>> build_layers(const ModelConfig& cfg, const TensorMap& weights);

// Compute workspace size needed per token across all layer types
int compute_workspace_per_token(const ModelConfig& cfg, const std::vector<std::unique_ptr<Layer>>& layers);

// Compute saved buffer size per token across all layer types
int compute_saved_per_token(const ModelConfig& cfg, const std::vector<std::unique_ptr<Layer>>& layers);
