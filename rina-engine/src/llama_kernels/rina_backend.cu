#include "common.cuh"
#include <cstdarg>
#include <memory>

static bool rina_dev_info_init = false;
static ggml_cuda_device_info rina_dev_info;

static void rina_init_device_info() {
    if (rina_dev_info_init) return;
    rina_dev_info_init = true;
    int count = 0;
    cudaGetDeviceCount(&count);
    rina_dev_info.device_count = (count > 0 && count <= GGML_CUDA_MAX_DEVICES) ? count : 1;
    for (int i = 0; i < rina_dev_info.device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto & d = rina_dev_info.devices[i];
        d.cc = prop.major * 100 + prop.minor * 10;
        d.nsm = prop.multiProcessorCount;
        d.smpb = prop.sharedMemPerBlock;
        d.smpbo = prop.sharedMemPerBlockOptin;
        d.integrated = prop.integrated;
        d.vmm = false;
        d.total_vram = prop.totalGlobalMem;
        d.warp_size = prop.warpSize;
        d.supports_cooperative_launch = prop.cooperativeLaunch;
    }
}

int64_t ggml_blck_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_IQ4_XS: return 256;
        default: return 1;
    }
}

size_t ggml_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q8_1: return sizeof(block_q8_1);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS: return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ3_S: return sizeof(block_iq3_s);
        case GGML_TYPE_IQ1_S: return sizeof(block_iq1_s);
        case GGML_TYPE_IQ1_M: return sizeof(block_iq1_m);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default: return 4;
    }
}

// ─── ggml backend stubs needed by mmq.cu ───
// common.cuh → ggml-cuda.h → ggml-backend.h provides the declarations.
// We provide the definitions here.

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    (void)buffer; (void)tensor;
    return 0;
}
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    (void)buffer;
    return GGML_BACKEND_BUFFER_USAGE_ANY;
}

size_t ggml_nbytes(const ggml_tensor * t) {
    if (!t) return 0;
    return (size_t)t->ne[0] * t->nb[0] / ggml_blck_size(t->type) * t->ne[1] * t->ne[2] * t->ne[3];
}
bool ggml_is_contiguously_allocated(const ggml_tensor *) { return true; }
size_t ggml_element_size(const ggml_tensor * t) { return ggml_type_size(t->type); }

const ggml_cuda_device_info & ggml_cuda_info() {
    rina_init_device_info();
    return rina_dev_info;
}

void ggml_cuda_set_device(int device) {
    cudaSetDevice(device);
}

int ggml_cuda_get_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

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

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::make_unique<rina_cuda_pool>();
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; j++) {
            pools[i][j].reset();
        }
    }
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "GGML_ABORT at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s: %s (%s:%d in %s)\n", stmt, msg, file, line, func);
}
