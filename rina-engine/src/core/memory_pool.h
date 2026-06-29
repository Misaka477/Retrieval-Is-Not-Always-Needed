#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <string>

// 逐层复用 workspace pool
// 前向时每层申请临时 buffer，推理完释放，pool 复用显存

struct MemoryPool {
    cudaStream_t stream;
    std::vector<std::pair<std::string, void*>> blocks;
    size_t total_bytes;

    void init(cudaStream_t s = 0) { stream = s; total_bytes = 0; }

    void* alloc(const std::string& name, size_t bytes) {
        // 查找是否已有同名 buffer
        for (auto& [n, p] : blocks)
            if (n == name) return p;
        // 找是否有大小够的 free block
        for (auto& [n, p] : blocks)
            if (n.empty() && bytes <= block_size(p)) { n = name; return p; }
        // 新分配
        void* ptr;
        cudaMallocAsync(&ptr, bytes, stream);
        blocks.push_back({name, ptr});
        total_bytes += bytes;
        return ptr;
    }

    void release(const std::string& name) {
        for (auto& [n, p] : blocks)
            if (n == name) { n = ""; return; }
    }

    void free_all() {
        for (auto& [n, p] : blocks)
            if (!p) cudaFree(p);
        blocks.clear();
        total_bytes = 0;
    }

private:
    size_t block_size(void* ptr) {
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, ptr);
        return 0; // 无法轻易获取分配大小, 简化处理
    }
};
