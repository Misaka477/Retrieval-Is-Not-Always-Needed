// RINA-specific ggml backend overrides.
// All standard ggml functions (ggml_type_size, ggml_blck_size, ggml_nbytes, etc.)
// come from the real ggml_core library. This file provides only RINA-specific
// customizations on top of the upstream ggml CUDA backend.

#include "common.cuh"
#include <memory>

// RINA custom CUDA pool allocator with direct cudaMalloc/cudaFree
struct rina_cuda_pool : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "rina_cuda_pool::alloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
            return nullptr;
        }
        if (actual_size) *actual_size = size;
        return ptr;
    }
    void free(void * ptr, size_t) override {
        if (ptr) cudaFree(ptr);
    }
};

// Override the pool factory to use RINA's pool
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::make_unique<rina_cuda_pool>();
}
