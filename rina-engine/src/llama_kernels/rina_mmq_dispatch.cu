#include "common.cuh"
#include "mmq.cuh"

int rina_get_mmq_x_max(int cc) {
    return get_mmq_x_max_host(cc);
}

void rina_mmq_dispatch(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case 2:  mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream); break;
        case 3:  mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream); break;
        case 6:  mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream); break;
        case 7:  mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream); break;
        case 8:  mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream); break;
        case 10: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream); break;
        case 11: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream); break;
        case 12: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream); break;
        case 13: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream); break;
        case 14: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream); break;
        case 16: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream); break;
        case 17: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream); break;
        case 18: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream); break;
        case 19: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream); break;
        case 23: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream); break;
        case 24: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream); break;
        default: break;
    }
}
