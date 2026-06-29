#pragma once
#include "quant.h"
#include "config.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <string>

struct WeightTensor {
    void*      data;
    int        shape[4];
    int        n_dim;
    QuantType  quant_type;
    int        n_elems;

    WeightTensor() : data(nullptr), n_dim(0), quant_type(QuantType::FP32), n_elems(0) {
        for (int i = 0; i < 4; i++) shape[i] = 0;
    }

    void to_device(const void* host_data, const TensorSpec& spec) {
        quant_type = static_cast<QuantType>(spec.quant_type);
        n_elems = 1; n_dim = spec.n_dim;
        for (int i = 0; i < n_dim; i++) { shape[i] = spec.shape[i]; n_elems *= spec.shape[i]; }
        size_t bytes = quantized_size(n_elems, quant_type);
        cudaMalloc(&data, bytes);
        cudaMemcpy(data, host_data, bytes, cudaMemcpyHostToDevice);
    }

    void free() { if (data) { cudaFree(data); data = nullptr; } }
};

struct TensorMap {
    std::unordered_map<std::string, WeightTensor> tensors;
    void add(const std::string& name, WeightTensor&& t) { tensors[name] = std::move(t); }
    const WeightTensor* get(const std::string& name) const {
        auto it = tensors.find(name); return it != tensors.end() ? &it->second : nullptr;
    }
    WeightTensor* get(const std::string& name) {
        auto it = tensors.find(name); return it != tensors.end() ? &it->second : nullptr;
    }
    void free_all() { for (auto& [n, t] : tensors) t.free(); tensors.clear(); }
};
