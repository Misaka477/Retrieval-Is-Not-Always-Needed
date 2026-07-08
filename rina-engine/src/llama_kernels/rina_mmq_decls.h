#pragma once
// Lightweight declarations that avoid including the 4176-line mmq.cuh.
// Include common.cuh (for ggml_backend_cuda_context, ggml_type) before this.
#include <cuda_runtime.h>
#include <cstdint>

struct mmq_args {
    const char * x; int type_x; const int * y;
    const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst;
    int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y;
    int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y;
    int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    bool use_stream_k; int64_t ncols_max;
};

struct ggml_backend_cuda_context;

int rina_get_mmq_x_max(int cc);
void rina_mmq_dispatch(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
