#pragma once
#include "core/config.h"
#include "core/tensor.h"

ModelConfig parse_config(const char* path);
void load_weights(const char* path, TensorMap& tensors);
bool load_model(const char* path, ModelConfig& cfg, TensorMap& tensors);

// HF direct loader: reads config.json + safetensors, quantizes on-the-fly
bool load_hf_model(const char* dir_path, ModelConfig& cfg, TensorMap& tensors,
                   int quant_bits = 4);

void model_forward_fp32(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream,
    int start_pos = 0);
float model_train(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream);

// v2: new Layer-based implementation
void model_forward_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream,
    int start_pos = 0);
float model_train_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream);
