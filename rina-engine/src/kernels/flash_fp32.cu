// FlashAttention v2-style tiled attention (fp32).
// One block per (bh, tile_row_of_q). Processes Br rows of Q at once.
// K and V are loaded in Bc-sized tiles and reused across all Q rows in the tile.
// Online softmax: incremental m (row max), l (row sum), O partial sums.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define Br 32
#define Bc 32

__global__ void flashattn_fwd_tiled(const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int Bh, int T, int dq, int dh) {
    extern __shared__ float smem[];
    float* K_smem = smem;                // [Bc, dq]
    float* V_smem = K_smem + Bc * dq;    // [Bc, dh]
    float* O_smem = V_smem + Bc * dh;    // [Br, dh] — partial output per Q row

    int bh = blockIdx.x;
    int q_start = blockIdx.y * Br;
    if (bh >= Bh || q_start >= T) return;
    int q_end = min(q_start + Br, T);
    int nq = q_end - q_start;

    const float* Qb = Q + bh * T * dq;
    const float* Kb = K + bh * T * dq;
    const float* Vb = V + bh * T * dh;
    float* Ob = O + bh * T * dh;

    // Zero O_smem
    for (int i = threadIdx.x; i < nq * dh; i += blockDim.x) O_smem[i] = 0.0f;

    // m and l for online softmax (per Q row)
    float m_row[Br], l_row[Br];
    for (int i = 0; i < nq; i++) { m_row[i] = -1e10f; l_row[i] = 0.0f; }

    for (int k_start = 0; k_start < T; k_start += Bc) {
        int k_end = min(k_start + Bc, T);
        int nk = k_end - k_start;

        // Load K_smem and V_smem
        for (int i = threadIdx.x; i < nk * dq; i += blockDim.x)
            K_smem[i] = Kb[(k_start + i / dq) * dq + i % dq];
        for (int i = threadIdx.x; i < nk * dh; i += blockDim.x)
            V_smem[i] = Vb[(k_start + i / dh) * dh + i % dh];
        __syncthreads();

        for (int i = 0; i < nq; i++) {
            int qi = q_start + i;
            if (qi >= T) break;
            float m = m_row[i], l = l_row[i];
            const float* q_ptr = Qb + qi * dq;

            for (int kj = 0; kj < nk; kj++) {
                int kj_global = k_start + kj;
                if (kj_global > qi) break;  // causal

                // Dot product
                float dot = 0.0f;
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    dot += q_ptr[d] * K_smem[kj * dq + d];
                for (int w = 16; w > 0; w >>= 1) dot += __shfl_xor_sync(0xFFFFFFFF, dot, w);

                if (threadIdx.x == 0) {
                    float s = dot * rsqrtf((float)dq);
                    float old_l = l;
                    float new_m = fmaxf(m, s);
                    l = l * expf(m - new_m) + expf(s - new_m);
                    m = new_m;

                    // Online softmax: O = O * (old_l * exp(old_m - m) / l) + exp(s - m) / l * V
                    float scale_old = old_l * expf(m_row[i] - m) / l;
                    float scale_new = expf(s - m) / l;
                    for (int d = 0; d < dh; d++)
                        O_smem[i * dh + d] = O_smem[i * dh + d] * scale_old
                                           + V_smem[kj * dh + d] * scale_new;
                    m_row[i] = m; l_row[i] = l;
                    // Note: m_row[i] here is the updated m (new_m), but when used
                    // as old_m in the next iteration, it represents max up to prev K row.
                }
            }
        }
        __syncthreads();
    }

    // Write result
    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        if (qi >= T) break;
        for (int d = threadIdx.x; d < dh; d += blockDim.x)
            Ob[qi * dh + d] = O_smem[i * dh + d];
    }
}

void launch_flash_attn_fp32(const float* Q, const float* K, const float* V, float* O,
                             int B, int H, int T, int dq, int dh, cudaStream_t stream) {
    int Bh = B * H;
    int grid_y = (T + Br - 1) / Br;
    int shmem = (Bc * dq + Bc * dh + Br * dh) * sizeof(float);
    if (shmem > 48 * 1024) {
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(flashattn_fwd_tiled,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
            attr_set = true;
        }
    }
    dim3 grid(Bh, grid_y);
    flashattn_fwd_tiled<<<grid, 32, shmem, stream>>>(Q, K, V, O, Bh, T, dq, dh);
}

// Transpose attention output from [H,T,dh] to [T,H*dh]
__global__ void transpose_attn_k(float* dst, const float* src, int H, int T, int dh) {
    int h = blockIdx.x * 16 + threadIdx.x;
    int t = blockIdx.y * 16 + threadIdx.y;
    if (h >= H || t >= T) return;
    for (int d = 0; d < dh; d++)
        dst[t * H * dh + h * dh + d] = src[h * T * dh + t * dh + d];
}

void launch_transpose_attn(float* dst, const float* src, int H, int T, int dh, cudaStream_t stream) {
    dim3 grid((H+15)/16, (T+15)/16);
    transpose_attn_k<<<grid, dim3(16,16), 0, stream>>>(dst, src, H, T, dh);
}

// Modified forward that also saves softmax stats (m, l) for backward
__global__ void flashattn_fwd_save_stats(const float* __restrict__ Q,
    const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float* __restrict__ m_out, float* __restrict__ l_out,
    int Bh, int T, int dq, int dh) {
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = K_smem + Bc * dq;
    float* O_smem = V_smem + Bc * dh;

    int bh = blockIdx.x;
    int q_start = blockIdx.y * Br;
    if (bh >= Bh || q_start >= T) return;
    int q_end = min(q_start + Br, T);
    int nq = q_end - q_start;

    const float* Qb = Q + bh * T * dq;
    const float* Kb = K + bh * T * dq;
    const float* Vb = V + bh * T * dh;
    float* Ob = O + bh * T * dh;
    float* mb = m_out + bh * T;
    float* lb = l_out + bh * T;

    for (int i = threadIdx.x; i < nq * dh; i += blockDim.x) O_smem[i] = 0.0f;

    float m_row[Br], l_row[Br];
    for (int i = 0; i < nq; i++) { m_row[i] = -1e10f; l_row[i] = 0.0f; }

    for (int k_start = 0; k_start < T; k_start += Bc) {
        int k_end = min(k_start + Bc, T);
        int nk = k_end - k_start;

        for (int i = threadIdx.x; i < nk * dq; i += blockDim.x)
            K_smem[i] = Kb[(k_start + i / dq) * dq + i % dq];
        for (int i = threadIdx.x; i < nk * dh; i += blockDim.x)
            V_smem[i] = Vb[(k_start + i / dh) * dh + i % dh];
        __syncthreads();

        float scale = rsqrtf((float)dq);
        for (int i = 0; i < nq; i++) {
            int qi = q_start + i;
            if (qi >= T) break;
            float m = m_row[i], l = l_row[i];
            const float* q_ptr = Qb + qi * dq;

            for (int kj = 0; kj < nk; kj++) {
                int kj_global = k_start + kj;
                if (kj_global > qi) break;

                float dot = 0.0f;
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    dot += q_ptr[d] * K_smem[kj * dq + d];
                for (int w = 16; w > 0; w >>= 1) dot += __shfl_xor_sync(0xFFFFFFFF, dot, w);

                if (threadIdx.x == 0) {
                    float s = dot * scale;
                    float old_l = l;
                    float new_m = fmaxf(m, s);
                    l = l * expf(m - new_m) + expf(s - new_m);
                    m = new_m;
                    float scale_old = old_l * expf(m_row[i] - m) / l;
                    float scale_new = expf(s - m) / l;
                    for (int d = 0; d < dh; d++)
                        O_smem[i * dh + d] = O_smem[i * dh + d] * scale_old
                                           + V_smem[kj * dh + d] * scale_new;
                    m_row[i] = m; l_row[i] = l;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        if (qi >= T) break;
        for (int d = threadIdx.x; d < dh; d += blockDim.x)
            Ob[qi * dh + d] = O_smem[i * dh + d];
        if (threadIdx.x == 0) {
            mb[qi] = m_row[i];
            lb[qi] = l_row[i];
        }
    }
}

void launch_flashattn_fwd_save_stats(const float* Q, const float* K,
    const float* V, float* O, float* m_out, float* l_out,
    int B, int H, int T, int dq, int dh, cudaStream_t stream) {
    int Bh = B * H;
    int grid_y = (T + Br - 1) / Br;
    int shmem = (Bc * dq + Bc * dh + Br * dh) * sizeof(float);
    dim3 grid(Bh, grid_y);
    flashattn_fwd_save_stats<<<grid, 32, shmem, stream>>>(Q, K, V, O, m_out, l_out, Bh, T, dq, dh);
}

// FlashAttention v2 backward (tiled, causal).
// D = row_sum(dO * O) computed upfront, then tile loops for dQ, dK, dV.
__global__ void flashattn_bwd_v2(const float* __restrict__ Q,
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ O, const float* __restrict__ dO,
    const float* __restrict__ m_saved, const float* __restrict__ l_saved,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    int Bh, int T, int dq, int dh) {
    extern __shared__ float smem[];
    float* K_smem = smem;             // [Bc, dq]
    float* V_smem = K_smem + Bc * dq; // [Bc, dh]

    int bh = blockIdx.x;
    int q_start = blockIdx.y * Br;
    if (bh >= Bh || q_start >= T) return;
    int q_end = min(q_start + Br, T);
    int nq = q_end - q_start;

    const float* Qb = Q + bh * T * dq;
    const float* Kb = K + bh * T * dq;
    const float* Vb = V + bh * T * dh;
    const float* Ob = O + bh * T * dh;
    const float* dOb = dO + bh * T * dh;
    const float* mb = m_saved + bh * T;
    const float* lb = l_saved + bh * T;
    float* dQb = dQ + bh * T * dq;
    float* dKb = dK + bh * T * dq;
    float* dVb = dV + bh * T * dh;

    float scale = rsqrtf((float)dq);

    // Step 1: D[qi] = row_sum(dO[qi] * O[qi]) for all Q rows in this tile
    float D_val[Br];
    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        float d = 0.0f;
        for (int j = threadIdx.x; j < dh; j += blockDim.x)
            d += dOb[qi * dh + j] * Ob[qi * dh + j];
        for (int w = 16; w > 0; w >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFF, d, w);
        D_val[i] = d;
    }

    // Zero local dK, dV accumulators (stored in global memory, atomicAdd for competition)
    // For dQ, write directly (each Q row belongs to exactly one Q tile)
    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        for (int d = threadIdx.x; d < dq; d += blockDim.x)
            dQb[qi * dq + d] = 0.0f;
    }

    // Step 2: Iterate over KV tiles
    for (int k_start = 0; k_start < T; k_start += Bc) {
        int k_end = min(k_start + Bc, T);
        int nk = k_end - k_start;

        // Load K, V for this tile
        for (int i = threadIdx.x; i < nk * dq; i += blockDim.x)
            K_smem[i] = Kb[(k_start + i / dq) * dq + i % dq];
        for (int i = threadIdx.x; i < nk * dh; i += blockDim.x)
            V_smem[i] = Vb[(k_start + i / dh) * dh + i % dh];
        __syncthreads();

        // For each Q row in this tile
        for (int i = 0; i < nq; i++) {
            int qi = q_start + i;
            float D_i = D_val[i];

            // For each K position that attends to this Q position (causal: k <= qi)
            for (int k = 0; k < nk; k++) {
                int k_global = k_start + k;
                if (k_global > qi) break;

                // Compute S = Q[qi] @ K[k_global]^T / sqrt(dq)
                float s = 0.0f;
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    s += Qb[qi * dq + d] * K_smem[k * dq + d];
                for (int w = 16; w > 0; w >>= 1)
                    s += __shfl_xor_sync(0xFFFFFFFF, s, w);
                s *= scale;

                // P = softmax = exp(S - m) / l
                float p = expf(s - mb[qi]) / (lb[qi] + 1e-10f);

                // dV += P^T @ dO
                for (int d = threadIdx.x; d < dh; d += blockDim.x)
                    atomicAdd(&dVb[k_global * dh + d], p * dOb[qi * dh + d]);

                // dP = dO[qi] @ V[k_global]^T  (scalar)
                float dp = 0.0f;
                for (int d = threadIdx.x; d < dh; d += blockDim.x)
                    dp += dOb[qi * dh + d] * V_smem[k * dh + d];
                for (int w = 16; w > 0; w >>= 1)
                    dp += __shfl_xor_sync(0xFFFFFFFF, dp, w);

                // dS = P * (dP - D)
                float ds = p * (dp - D_i);

                // dQ[qi] += dS @ K[k_global] / sqrt(dq)
                // No atomic needed: each q_start belongs to exactly one block
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    dQb[qi * dq + d] += ds * K_smem[k * dq + d] * scale;

                // dK[k_global] += dS @ Q[qi] / sqrt(dq)
                // Atomic needed: multiple Q tiles may write to same K position
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    atomicAdd(&dKb[k_global * dq + d], ds * Qb[qi * dq + d] * scale);
            }
        }
        __syncthreads();
    }
}

void launch_flash_attn_bwd_fp32(const float* Q, const float* K,
    const float* V, const float* O, const float* dO,
    const float* m_saved, const float* l_saved,
    float* dQ, float* dK, float* dV,
    int B, int H, int T, int dq, int dh, cudaStream_t stream) {
    int Bh = B * H;
    int grid_y = (T + Br - 1) / Br;
    int shmem = (Bc * dq + Bc * dh) * sizeof(float);
    dim3 grid(Bh, grid_y);
    flashattn_bwd_v2<<<grid, 32, shmem, stream>>>(Q, K, V, O, dO, m_saved, l_saved,
        dQ, dK, dV, Bh, T, dq, dh);
}

// ════════════════════════════════════════════════════════════════
// bf16 FlashAttention forward (no stats saved, for inference)
// ════════════════════════════════════════════════════════════════

#define Br_bf 32
#define Bc_bf 32

__global__ void flashattn_fwd_tiled_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int Bh, int T, int dq, int dh) {
    extern __shared__ __nv_bfloat16 smem_bf[];
    __nv_bfloat16* K_smem = smem_bf;
    __nv_bfloat16* V_smem = K_smem + Bc_bf * dq;
    __nv_bfloat16* O_smem = V_smem + Bc_bf * dh;

    int bh = blockIdx.x;
    int q_start = blockIdx.y * Br_bf;
    if (bh >= Bh || q_start >= T) return;
    int q_end = min(q_start + Br_bf, T);
    int nq = q_end - q_start;

    const __nv_bfloat16* Qb = Q + (size_t)bh * T * dq;
    const __nv_bfloat16* Kb = K + (size_t)bh * T * dq;
    const __nv_bfloat16* Vb = V + (size_t)bh * T * dh;
    __nv_bfloat16* Ob = O + (size_t)bh * T * dh;

    for (int i = threadIdx.x; i < nq * dh; i += blockDim.x) O_smem[i] = __float2bfloat16(0.0f);

    float m_row[Br_bf], l_row[Br_bf];
    for (int i = 0; i < nq; i++) { m_row[i] = -1e10f; l_row[i] = 0.0f; }

    for (int k_start = 0; k_start < T; k_start += Bc_bf) {
        int k_end = min(k_start + Bc_bf, T);
        int nk = k_end - k_start;

        for (int i = threadIdx.x; i < nk * dq; i += blockDim.x)
            K_smem[i] = Kb[(size_t)(k_start + i / dq) * dq + i % dq];
        for (int i = threadIdx.x; i < nk * dh; i += blockDim.x)
            V_smem[i] = Vb[(size_t)(k_start + i / dh) * dh + i % dh];
        __syncthreads();

        for (int i = 0; i < nq; i++) {
            int qi = q_start + i;
            if (qi >= T) break;
            float m = m_row[i], l = l_row[i];
            const __nv_bfloat16* q_ptr = Qb + (size_t)qi * dq;

            for (int kj = 0; kj < nk; kj++) {
                int kj_global = k_start + kj;
                if (kj_global > qi) break;

                float dot = 0.0f;
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    dot += __bfloat162float(q_ptr[d]) * __bfloat162float(K_smem[(size_t)kj * dq + d]);
                for (int w = 16; w > 0; w >>= 1) dot += __shfl_xor_sync(0xFFFFFFFF, dot, w);

                if (threadIdx.x == 0) {
                    float s = dot * rsqrtf((float)dq);
                    float old_l = l;
                    float new_m = fmaxf(m, s);
                    l = l * expf(m - new_m) + expf(s - new_m);
                    m = new_m;

                    float scale_old = old_l * expf(m_row[i] - m) / l;
                    float scale_new = expf(s - m) / l;
                    for (int d = 0; d < dh; d++)
                        O_smem[(size_t)i * dh + d] = __float2bfloat16(
                            __bfloat162float(O_smem[(size_t)i * dh + d]) * scale_old
                            + __bfloat162float(V_smem[(size_t)kj * dh + d]) * scale_new);
                    m_row[i] = m; l_row[i] = l;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        if (qi >= T) break;
        for (int d = threadIdx.x; d < dh; d += blockDim.x)
            Ob[(size_t)qi * dh + d] = O_smem[(size_t)i * dh + d];
    }
}

void launch_flash_attn_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O,
    int B, int H, int T, int dq, int dh, cudaStream_t stream) {
    int Bh = B * H;
    int grid_y = (T + Br_bf - 1) / Br_bf;
    int shmem = (Bc_bf * dq + Bc_bf * dh + Br_bf * dh) * sizeof(__nv_bfloat16);
    dim3 grid(Bh, grid_y);
    flashattn_fwd_tiled_bf16<<<grid, 32, shmem, stream>>>(Q, K, V, O, Bh, T, dq, dh);
}

// bf16 FlashAttention with stats saved (for training backward)
__global__ void flashattn_fwd_save_stats_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ m_out, float* __restrict__ l_out,
    int Bh, int T, int dq, int dh) {
    extern __shared__ __nv_bfloat16 smem_bf[];
    __nv_bfloat16* K_smem = smem_bf;
    __nv_bfloat16* V_smem = K_smem + Bc_bf * dq;
    __nv_bfloat16* O_smem = V_smem + Bc_bf * dh;

    int bh = blockIdx.x;
    int q_start = blockIdx.y * Br_bf;
    if (bh >= Bh || q_start >= T) return;
    int q_end = min(q_start + Br_bf, T);
    int nq = q_end - q_start;

    const __nv_bfloat16* Qb = Q + (size_t)bh * T * dq;
    const __nv_bfloat16* Kb = K + (size_t)bh * T * dq;
    const __nv_bfloat16* Vb = V + (size_t)bh * T * dh;
    __nv_bfloat16* Ob = O + (size_t)bh * T * dh;
    float* mb = m_out + (size_t)bh * T;
    float* lb = l_out + (size_t)bh * T;

    for (int i = threadIdx.x; i < nq * dh; i += blockDim.x) O_smem[i] = __float2bfloat16(0.0f);

    float m_row[Br_bf], l_row[Br_bf];
    for (int i = 0; i < nq; i++) { m_row[i] = -1e10f; l_row[i] = 0.0f; }

    for (int k_start = 0; k_start < T; k_start += Bc_bf) {
        int k_end = min(k_start + Bc_bf, T);
        int nk = k_end - k_start;

        for (int i = threadIdx.x; i < nk * dq; i += blockDim.x)
            K_smem[i] = Kb[(size_t)(k_start + i / dq) * dq + i % dq];
        for (int i = threadIdx.x; i < nk * dh; i += blockDim.x)
            V_smem[i] = Vb[(size_t)(k_start + i / dh) * dh + i % dh];
        __syncthreads();

        float scale = rsqrtf((float)dq);
        for (int i = 0; i < nq; i++) {
            int qi = q_start + i;
            if (qi >= T) break;
            float m = m_row[i], l = l_row[i];
            const __nv_bfloat16* q_ptr = Qb + (size_t)qi * dq;

            for (int kj = 0; kj < nk; kj++) {
                int kj_global = k_start + kj;
                if (kj_global > qi) break;

                float dot = 0.0f;
                for (int d = threadIdx.x; d < dq; d += blockDim.x)
                    dot += __bfloat162float(q_ptr[d]) * __bfloat162float(K_smem[(size_t)kj * dq + d]);
                for (int w = 16; w > 0; w >>= 1) dot += __shfl_xor_sync(0xFFFFFFFF, dot, w);

                if (threadIdx.x == 0) {
                    float s = dot * scale;
                    float old_l = l;
                    float new_m = fmaxf(m, s);
                    l = l * expf(m - new_m) + expf(s - new_m);
                    m = new_m;
                    float scale_old = old_l * expf(m_row[i] - m) / l;
                    float scale_new = expf(s - m) / l;
                    for (int d = 0; d < dh; d++)
                        O_smem[(size_t)i * dh + d] = __float2bfloat16(
                            __bfloat162float(O_smem[(size_t)i * dh + d]) * scale_old
                            + __bfloat162float(V_smem[(size_t)kj * dh + d]) * scale_new);
                    m_row[i] = m; l_row[i] = l;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < nq; i++) {
        int qi = q_start + i;
        if (qi >= T) break;
        for (int d = threadIdx.x; d < dh; d += blockDim.x)
            Ob[(size_t)qi * dh + d] = O_smem[(size_t)i * dh + d];
        if (threadIdx.x == 0) {
            mb[qi] = m_row[i];
            lb[qi] = l_row[i];
        }
    }
}

void launch_flashattn_fwd_save_stats_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K,
    const __nv_bfloat16* V, __nv_bfloat16* O,
    float* m_out, float* l_out,
    int B, int H, int T, int dq, int dh, cudaStream_t stream) {
    int Bh = B * H;
    int grid_y = (T + Br_bf - 1) / Br_bf;
    int shmem = (Bc_bf * dq + Bc_bf * dh + Br_bf * dh) * sizeof(__nv_bfloat16);
    dim3 grid(Bh, grid_y);
    flashattn_fwd_save_stats_bf16<<<grid, 32, shmem, stream>>>(
        Q, K, V, O, m_out, l_out, Bh, T, dq, dh);
}

// bf16 transpose attention output: [H,T,dh] → [T,H*dh]
__global__ void transpose_attn_bf16_k(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int H, int T, int dh) {
    int h = blockIdx.x * 16 + threadIdx.x;
    int t = blockIdx.y * 16 + threadIdx.y;
    if (h >= H || t >= T) return;
    for (int d = 0; d < dh; d++)
        dst[(size_t)t * H * dh + (size_t)h * dh + d] = src[(size_t)h * T * dh + (size_t)t * dh + d];
}

void launch_transpose_attn_bf16(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int H, int T, int dh, cudaStream_t stream) {
    dim3 grid((H + 15) / 16, (T + 15) / 16);
    transpose_attn_bf16_k<<<grid, dim3(16, 16), 0, stream>>>(dst, src, H, T, dh);
}
