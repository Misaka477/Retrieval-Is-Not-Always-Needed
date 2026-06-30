#pragma once
#include "core/config.h"
#include "core/tensor.h"

ModelConfig parse_config(const char* path);
void load_weights(const char* path, TensorMap& tensors);
bool load_model(const char* path, ModelConfig& cfg, TensorMap& tensors);

// v1: inline arch code (current, to be replaced)
void model_forward_direct(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream);
void model_forward_fp32(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream);
float model_train(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream);

// v2: new Layer-based implementation
void model_forward_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream);
float model_train_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, const int* targets, float* loss_d,
    int B, int T, int step, cudaStream_t stream);
