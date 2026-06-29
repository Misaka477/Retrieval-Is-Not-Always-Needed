#include <cuda_runtime.h>
#include <cmath>

// Cross-warp helper: all 32 threads in warp 0 must participate in __shfl
// nwarps = number of warps (<= 32 for up to 1024 threads)
#define CW_REDUCE_MAX(v, smem, nwarps) do {\
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = v;\
    __syncthreads();\
    if (threadIdx.x < 32) {\
        float _v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -1e10f;\
        for (int _w = nwarps / 2; _w > 0; _w >>= 1)\
            _v = fmaxf(_v, __shfl_xor_sync(0xFFFFFFFF, _v, _w));\
        if (threadIdx.x == 0) smem[0] = _v;\
    }\
    __syncthreads(); v = smem[0];\
} while(0)
#define CW_REDUCE_SUM(v, smem, nwarps) do {\
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = v;\
    __syncthreads();\
    if (threadIdx.x < 32) {\
        float _v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;\
        for (int _w = nwarps / 2; _w > 0; _w >>= 1)\
            _v += __shfl_xor_sync(0xFFFFFFFF, _v, _w);\
        if (threadIdx.x == 0) smem[0] = _v;\
    }\
    __syncthreads(); v = smem[0];\
} while(0)

// dlogits[B*T, V] = softmax(logits) - one_hot(targets), with mean reduction
__global__ void crossentropy_dlogits_kernel(const float* logits,
    const int* targets, float* dlogits, int N, int V) {
    extern __shared__ float smem[];
    int i = blockIdx.x;
    if (i >= N) return;
    int nwarps = (blockDim.x + 31) / 32;

    const float* row = logits + i * V;
    float* drow = dlogits + i * V;

    // Phase 1: max reduction
    float mx = -1e10f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
        mx = fmaxf(mx, row[j]);
    for (int w = 16; w > 0; w >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, w));
    CW_REDUCE_MAX(mx, smem, nwarps);

    // Phase 2: sum of exps
    float sum_exp = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float e = expf(row[j] - mx);
        drow[j] = e;
        sum_exp += e;
    }
    for (int w = 16; w > 0; w >>= 1)
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, w);
    CW_REDUCE_SUM(sum_exp, smem, nwarps);

    // Phase 3: normalize and compute gradient
    int t = targets[i];
    float inv_N = 1.0f / N;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float p = drow[j] / sum_exp;
        drow[j] = (p - (j == t ? 1.0f : 0.0f)) * inv_N;
    }
}

// loss = mean(-log(softmax(target)))
__global__ void crossentropy_loss_kernel(const float* logits,
    const int* targets, float* loss_buf, int N, int V) {
    extern __shared__ float smem[];
    int i = blockIdx.x;
    if (i >= N) return;
    int nwarps = (blockDim.x + 31) / 32;

    const float* row = logits + i * V;
    float mx = -1e10f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
        mx = fmaxf(mx, row[j]);
    for (int w = 16; w > 0; w >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, w));
    CW_REDUCE_MAX(mx, smem, nwarps);

    float sum_exp = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
        sum_exp += expf(row[j] - mx);
    for (int w = 16; w > 0; w >>= 1)
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, w);
    CW_REDUCE_SUM(sum_exp, smem, nwarps);

    int t = targets[i];
    float loss_i = -logf(fmaxf(expf(row[t] - mx) / sum_exp, 1e-10f));
    if (threadIdx.x == 0) loss_buf[i] = loss_i;
}

// Simple grid-stride sum reduction
__global__ void sum_reduce_f32(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        s += in[i];
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) smem[threadIdx.x] += smem[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

float launch_crossentropy_fp32(const float* logits, const int* targets,
    float* dlogits, int N, int V, cudaStream_t stream) {
    int threads = V < 256 ? (V < 128 ? 64 : 128) : 256;
    int nwarps = (threads + 31) / 32;
    int shmem = nwarps * sizeof(float);

    // Gradient: fill dlogits
    crossentropy_dlogits_kernel<<<N, threads, shmem, stream>>>(logits, targets, dlogits, N, V);

    // Loss: per-sample losses → sum → scalar
    float *d_loss_buf, *d_loss_part;
    cudaMallocAsync(&d_loss_buf, N * sizeof(float), stream);
    int nparts = (N + 255) / 256;
    cudaMallocAsync(&d_loss_part, nparts * sizeof(float), stream);

    crossentropy_loss_kernel<<<N, threads, shmem, stream>>>(logits, targets, d_loss_buf, N, V);

    sum_reduce_f32<<<nparts, 256, 256 * sizeof(float), stream>>>(d_loss_buf, d_loss_part, N);
    if (nparts > 1) {
        float *d_loss_scalar;
        cudaMallocAsync(&d_loss_scalar, sizeof(float), stream);
        sum_reduce_f32<<<1, 256, 256 * sizeof(float), stream>>>(d_loss_part, d_loss_scalar, nparts);
        float loss;
        cudaMemcpyAsync(&loss, d_loss_scalar, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFreeAsync(d_loss_scalar, stream);
        cudaFreeAsync(d_loss_buf, stream);
        cudaFreeAsync(d_loss_part, stream);
        return loss / N;
    }
    float loss;
    cudaMemcpyAsync(&loss, d_loss_part, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_loss_buf, stream);
    cudaFreeAsync(d_loss_part, stream);
    return loss / N;
}
