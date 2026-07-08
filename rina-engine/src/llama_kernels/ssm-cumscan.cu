#include "ssm-cumscan.cuh"

// InertiaWave SSM cumscan: log-space cumulative sum over sequence dimension
// mem: [dh, H, T, B]  — memory values per (head, timestep)
// decay: [1, H, T, B] — decay factor per (head, timestep), broadcast across dh
// output: [dh, H, T, B] — same shape as mem
//
// For each (b, h, d):
//   lcs = 0
//   wcs = 0
//   for t in 0..T-1:
//     dv = decay[0, h, t, b]
//     if dv <= 1e-38: dv = 1e-38
//     lcs += log(dv)
//     ca = exp(lcs)
//     wcs += mem[d, h, t, b] / (ca + 1e-30)
//     out[d, h, t, b] = ca * wcs

__global__ void ssm_cumscan_f32(
    const float * __restrict__ mem,
    const float * __restrict__ decay,
    float * __restrict__ dst,
    int dh, int H, int T, int B) {

    int bh = blockIdx.x;   // batch*H + head
    int d  = threadIdx.x;  // dh dimension
    if (bh >= B * H || d >= dh) return;

    int b = bh / H;
    int h = bh % H;

    float lcs = 0.0f;
    float wcs = 0.0f;

    for (int t = 0; t < T; t++) {
        // decay: [1, H, T, B] — ne[0]=1, ne[1]=H, ne[2]=T, ne[3]=B
        // stride decay[ne[2]] = H*1 = H elements for one time step
        int decay_idx = h + t * H + b * T * H;
        float dv = decay[decay_idx];
        if (dv <= 1e-38f) dv = 1e-38f;
        lcs += logf(dv);
        float ca = expf(lcs);

        // mem: [dh, H, T, B]
        // position (d, h, t, b): offset = d + h*dh + t*H*dh + b*T*H*dh
        int mem_idx = d + h * dh + t * H * dh + b * T * H * dh;
        float m = mem[mem_idx];

        wcs += m / (ca + 1e-30f);

        int out_idx = mem_idx;
        dst[out_idx] = ca * wcs;
    }
}

void ggml_cuda_op_ssm_cumscan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // mem
    const ggml_tensor * src1 = dst->src[1]; // decay

    const int64_t dh = src0->ne[0];
    const int64_t H  = src0->ne[1];
    const int64_t T  = src0->ne[2];
    const int64_t B  = src0->ne[3];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int threads = (dh > 256) ? 256 : dh;
    const dim3 blocks(B * H, 1, 1);

    ssm_cumscan_f32<<<blocks, threads, 0, stream>>>(src0_d, src1_d, dst_d, dh, H, T, B);
}
