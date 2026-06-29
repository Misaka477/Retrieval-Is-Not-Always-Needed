#pragma once

struct Layer;
struct ModelConfig;
struct TensorMap;

// Factory entry point (defined in gqa_layer.cu)
Layer create_gqa_layer();
