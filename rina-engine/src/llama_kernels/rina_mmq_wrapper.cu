// Thin wrapper: calls into the real ggml_cuda library for quantized matmul.
// Does not include mmq.cuh directly — that would cause duplicate symbols
// with the ggml_cuda library.

#include "common.cuh"
#include "ggml-backend.h"

// Declared in mmq.cuh (compiled as part of ggml_cuda library)
void ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

void rina_launch_mmq(
    const void * weight_data, int ggml_type_id,
    const float * input, float * output,
    int M, int N, int K,
    cudaStream_t stream) {

    cudaGetLastError(); // clear pending errors

    ggml_backend_cuda_context ctx(ggml_cuda_get_device());
    ctx.streams[ctx.device][ctx.curr_stream_no] = stream;

    ggml_type wtype = (ggml_type)ggml_type_id;
    int bs = ggml_blck_size(wtype);
    size_t ts0 = ggml_type_size(wtype);

    ggml_tensor src0_stub = {};
    src0_stub.type   = wtype;
    src0_stub.ne[0]  = K;
    src0_stub.ne[1]  = N;
    src0_stub.ne[2]  = 1;
    src0_stub.ne[3]  = 1;
    src0_stub.nb[0]  = ts0;
    src0_stub.nb[1]  = (K / bs) * ts0;
    src0_stub.nb[2]  = src0_stub.nb[1] * N;
    src0_stub.nb[3]  = src0_stub.nb[2];
    src0_stub.data   = (void*)weight_data;

    ggml_tensor src1_stub = {};
    src1_stub.type   = GGML_TYPE_F32;
    src1_stub.ne[0]  = K;
    src1_stub.ne[1]  = M;
    src1_stub.ne[2]  = 1;
    src1_stub.ne[3]  = 1;
    src1_stub.nb[0]  = 4;
    src1_stub.nb[1]  = K * 4;
    src1_stub.nb[2]  = K * M * 4;
    src1_stub.nb[3]  = K * M * 4;
    src1_stub.data   = (void*)input;

    ggml_tensor dst_stub = {};
    dst_stub.type   = GGML_TYPE_F32;
    dst_stub.ne[0]  = N;
    dst_stub.ne[1]  = M;
    dst_stub.ne[2]  = 1;
    dst_stub.ne[3]  = 1;
    dst_stub.nb[0]  = 4;
    dst_stub.nb[1]  = N * 4;
    dst_stub.nb[2]  = N * M * 4;
    dst_stub.nb[3]  = N * M * 4;
    dst_stub.data   = output;

    ggml_cuda_mul_mat_q(ctx, &src0_stub, &src1_stub, nullptr, &dst_stub);
}

void rina_launch_mmvq(
    const void * weight_data, int ggml_type_id,
    const float * input, float * output,
    int N, int K,
    cudaStream_t stream) {
    rina_launch_mmq(weight_data, ggml_type_id, input, output, 1, N, K, stream);
}
