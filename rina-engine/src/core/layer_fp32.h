#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>

struct ModelConfig;
struct TensorMap;

// FP32 Layer interface — 对应 model_fp32.cu 的层处理
struct LayerFP32 {
    virtual ~LayerFP32() = default;
    virtual bool init(const ModelConfig& cfg, const TensorMap& weights, int layer_idx) = 0;
    // forward: path 部分（LN1 之后，residual 之前）
    //   path_in:  [B×T×dim] LN1 输出（output 写入此缓冲区）
    //   residual: [B×T×dim] 残差原值（用于 concat 等）
    //   ws:   workpace 缓冲区（大小由 workspace_per_token 决定）
    virtual void forward(const float* path_in, const float* residual, float* ws,
                         int B, int T, cudaStream_t stream) = 0;
    virtual int workspace_per_token(int dim, int n_heads, int head_dim) const = 0;
};

// 工厂注册
using LayerFP32Factory = LayerFP32* (*)();
void register_layer_fp32(const std::string& type, LayerFP32Factory f);
LayerFP32* create_layer_fp32(const std::string& type);
void init_builtin_layers_fp32();
