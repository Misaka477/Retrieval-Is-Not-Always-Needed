#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

extern "C" {

__global__ void cann_step_kernel(
    const float* h_in, const float* x, const float* patterns,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    float* h_out, int d_model, int n_patterns, float beta
) {
    int d = threadIdx.x, d2 = d_model * 2;
    extern __shared__ float smem[];
    float* h_shared = &smem[0];
    float* scores = &smem[d_model];
    if (d < d_model) {
        float a_s = 0, b_s = 0, xp = 0;
        for (int k = 0; k < d_model; k++) { a_s += w_a[d * d2 + k] * h_in[k]; b_s += w_b[d * d2 + k] * h_in[k]; }
        for (int k = 0; k < d_model; k++) { a_s += w_a[d * d2 + d_model + k] * x[k]; b_s += w_b[d * d2 + d_model + k] * x[k]; xp += w_p[d * d_model + k] * x[k]; }
        a_s += b_a[d]; b_s += b_b[d]; xp += b_p[d];
        float a_v = 1.0f / (1.0f + expf(-a_s)), b_v = 1.0f / (1.0f + expf(-b_s));
        h_shared[d] = a_v * h_in[d] + b_v * xp;
    }
    __syncthreads();
    if (d < n_patterns) {
        float sc = 0;
        for (int k = 0; k < d_model; k++) sc += patterns[d * d_model + k] * h_shared[k];
        scores[d] = sc * beta;
    }
    __syncthreads();
    if (d == 0) {
        float mx = -FLT_MAX;
        for (int i = 0; i < n_patterns; i++) if (scores[i] > mx) mx = scores[i];
        float se = 0;
        for (int i = 0; i < n_patterns; i++) { scores[i] = expf(scores[i] - mx); se += scores[i]; }
        for (int i = 0; i < n_patterns; i++) scores[i] /= se;
    }
    __syncthreads();
    if (d < d_model) {
        float at = 0; for (int i = 0; i < n_patterns; i++) at += patterns[i * d_model + d] * scores[i];
        float al = 0;
        for (int k = 0; k < d_model; k++) al += w_g[d * d2 + k] * h_in[k];
        for (int k = 0; k < d_model; k++) al += w_g[d * d2 + d_model + k] * x[k];
        al += b_g[d];
        float av = 1.0f / (1.0f + expf(-al));
        float hn = h_shared[d] + av * (at - h_shared[d]);
        smem[d] = hn; __syncthreads();
        if (d == 0) {
            float m = 0; for (int i = 0; i < d_model; i++) m += smem[i]; m /= d_model;
            float v = 0; for (int i = 0; i < d_model; i++) { float df = smem[i] - m; v += df * df; } v /= d_model;
            float is = rsqrtf(v + 1e-5f);
            for (int i = 0; i < d_model; i++) smem[i] = w_n[i] * (smem[i] - m) * is + b_n[i];
        }
        __syncthreads();
        h_out[d] = smem[d];
    }
}


__global__ void cann_sequence_kernel_v2(
    const float* h_init, const float* emb, const int* x_tokens,
    const float* patterns, const float* slot_table,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n, const float* sn_w, const float* sn_b,
    const float* head_w, const float* head_b,
    float* logits_out,
    float* saved_h_ssm, float* saved_gate_a, float* saved_gate_b,
    float* saved_alpha, float* saved_attn, float* saved_h_new,
    float* saved_cmean, float* saved_cinv_std,
    float* saved_hmean, float* saved_hinv_std,
    int batch, int seq_len, int d_model, int n_patterns, int vocab_size,
    float beta, int attract_every
) {
    int b = blockIdx.x, d = threadIdx.x, d2 = d_model * 2;
    extern __shared__ float smem[];
    float* state      = &smem[0];
    float* hbuf       = &smem[d_model];
    float* scores     = &smem[2 * d_model];
    float* state_buf  = &smem[2 * d_model + n_patterns];

    if (d < d_model) state[d] = h_init[b * d_model + d];
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        int b_t = b * seq_len + t;
        int emb_off = b * seq_len * d_model + t * d_model;
        int do_full = (attract_every < 1) || (t % attract_every == attract_every - 1);

        if (t == seq_len - 1 && d < d_model) {
            int tid = x_tokens[b_t];
            state[d] += slot_table[tid * d_model + d];
        }
        __syncthreads();

        float av = 0.0f, bv = 0.0f;
        if (d < d_model) {
            const float* x = &emb[emb_off];
            float a = 0, bb = 0, xp = 0;
            for (int k = 0; k < d_model; k++) { a += w_a[d * d2 + k] * state[k]; bb += w_b[d * d2 + k] * state[k]; }
            for (int k = 0; k < d_model; k++) { a += w_a[d * d2 + d_model + k] * x[k]; bb += w_b[d * d2 + d_model + k] * x[k]; xp += w_p[d * d_model + k] * x[k]; }
            a += b_a[d]; bb += b_b[d]; xp += b_p[d];
            av = 1.0f / (1.0f + expf(-a));
            bv = 1.0f / (1.0f + expf(-bb));
            hbuf[d] = av * state[d] + bv * xp;
        }

        if (d < d_model) {
            int save_off = b * seq_len * d_model + t * d_model + d;
            saved_gate_a[save_off] = av;
            saved_gate_b[save_off] = bv;
            saved_h_ssm[save_off] = hbuf[d];
        }
        __syncthreads();

        float alv = 0.0f;
        if (do_full) {
            if (d < n_patterns) {
                float s = 0;
                for (int k = 0; k < d_model; k++) s += patterns[d * d_model + k] * hbuf[k];
                scores[d] = s * beta;
            }
            __syncthreads();

            if (d == 0) {
                float mx = -FLT_MAX; for (int i = 0; i < n_patterns; i++) if (scores[i] > mx) mx = scores[i];
                float se = 0; for (int i = 0; i < n_patterns; i++) { scores[i] = expf(scores[i] - mx); se += scores[i]; }
                for (int i = 0; i < n_patterns; i++) scores[i] /= se;
            }
            __syncthreads();

            if (d < n_patterns) {
                int save_off = b * seq_len * n_patterns + t * n_patterns + d;
                saved_attn[save_off] = scores[d];
            }
            __syncthreads();

            if (d < d_model) {
                const float* x = &emb[emb_off];
                float at = 0; for (int i = 0; i < n_patterns; i++) at += patterns[i * d_model + d] * scores[i];
                float al = 0; for (int k = 0; k < d_model; k++) al += w_g[d * d2 + k] * state[k];
                for (int k = 0; k < d_model; k++) al += w_g[d * d2 + d_model + k] * x[k];
                al += b_g[d];
                alv = 1.0f / (1.0f + expf(-al));
                float hn = hbuf[d] + alv * (at - hbuf[d]);
                state_buf[d] = hn;
            }

            if (d < d_model) {
                int save_off = b * seq_len * d_model + t * d_model + d;
                saved_alpha[save_off] = alv;
            }
        } else {
            if (d < n_patterns) {
                int save_off = b * seq_len * n_patterns + t * n_patterns + d;
                saved_attn[save_off] = 0.0f;
            }
            if (d < d_model) {
                state_buf[d] = hbuf[d];
                int save_off = b * seq_len * d_model + t * d_model + d;
                saved_alpha[save_off] = 0.0f;
            }
        }
        __syncthreads();

        if (d < d_model) {
            int save_off = b * seq_len * d_model + t * d_model + d;
            saved_h_new[save_off] = state_buf[d];
            state[d] = state_buf[d];
        }
        __syncthreads();

        if (d == 0) {
            float m = 0; for (int i = 0; i < d_model; i++) m += state[i]; m /= (float)d_model;
            float v = 0; for (int i = 0; i < d_model; i++) { float df = state[i] - m; v += df * df; } v /= (float)d_model;
            float is = rsqrtf(v + 1e-5f);
            saved_cmean[b_t] = m;
            saved_cinv_std[b_t] = is;
            for (int i = 0; i < d_model; i++) state[i] = w_n[i] * (state[i] - m) * is + b_n[i];
        }
        __syncthreads();

        if (d == 0) {
            float m = 0; for (int i = 0; i < d_model; i++) m += state[i]; m /= (float)d_model;
            float v = 0; for (int i = 0; i < d_model; i++) { float df = state[i] - m; v += df * df; } v /= (float)d_model;
            float is = rsqrtf(v + 1e-5f);
            saved_hmean[b_t] = m;
            saved_hinv_std[b_t] = is;
            for (int i = 0; i < d_model; i++) hbuf[i] = sn_w[i] * (state[i] - m) * is + sn_b[i];
        }
        __syncthreads();

        if (d < vocab_size) {
            float lg = 0;
            for (int k = 0; k < d_model; k++) lg += head_w[d * d_model + k] * hbuf[k];
            lg += head_b[d];
            int lo = b * seq_len * vocab_size + t * vocab_size + d;
            logits_out[lo] = lg;
        }
        __syncthreads();
    }
}


__global__ void cann_sequence_backward_kernel(
    const float* grad_logits,
    const float* saved_h_ssm, const float* saved_gate_a,
    const float* saved_gate_b, const float* saved_alpha, const float* saved_attn,
    const float* saved_h_new, const float* saved_cmean, const float* saved_cinv_std,
    const float* saved_hmean, const float* saved_hinv_std,
    const float* emb, const int* x_tokens, const float* patterns,
    const float* slot_table,
    const float* h_init_fwd,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    const float* head_w, const float* head_b,
    const float* sn_w, const float* sn_b,
    float* d_h_init, float* d_emb, float* d_patterns,
    float* d_wa, float* d_ba, float* d_wb, float* d_bb,
    float* d_wg, float* d_bg, float* d_wp, float* d_bp,
    float* d_wn, float* d_bn,
    float* d_head_w, float* d_head_b,
    float* d_sn_w, float* d_sn_b,
    float* d_slot_table,
    float* scratch_d_attn, float* scratch_d_scores,
    int batch, int seq_len, int d_model, int n_patterns, int vocab_size,
    float beta, int attract_every
) {
    int b = blockIdx.x, d = threadIdx.x;
    if (d >= d_model) return;
    int d2 = d_model * 2;

    extern __shared__ float smem[];
    float* sh_h_prev   = &smem[0 * d_model];
    float* sh_emb_t    = &smem[1 * d_model];
    float* sh_t_a      = &smem[2 * d_model];
    float* sh_t_b      = &smem[3 * d_model];
    float* sh_t_c      = &smem[4 * d_model];
    float* sh_t_d      = &smem[5 * d_model];
    float* sh_d_attn   = &smem[6 * d_model];

    float d_h_next = 0.0f;
    float wn_d = w_n[d];
    float bn_d = b_n[d];
    float snw_d = sn_w[d];
    float snb_d = sn_b[d];

    for (int t = seq_len - 1; t >= 0; t--) {
        int b_t = b * seq_len + t;
        int t_off_dm = b * seq_len * d_model + t * d_model;
        int t_off_np = b * seq_len * n_patterns + t * n_patterns;
        int boff_logits = b * seq_len * vocab_size + t * vocab_size;
        int do_full = (attract_every < 1) || (t % attract_every == attract_every - 1);

        float h_new_d = saved_h_new[t_off_dm + d];
        float gate_a_d = saved_gate_a[t_off_dm + d];
        float gate_b_d = saved_gate_b[t_off_dm + d];
        float alpha_d = saved_alpha[t_off_dm + d];
        float h_ssm_d = saved_h_ssm[t_off_dm + d];
        float cmean = saved_cmean[b_t];
        float cinv = saved_cinv_std[b_t];
        float emb_d = emb[t_off_dm + d];

        float d_hn = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float gl = grad_logits[boff_logits + v];
            if (gl != 0.0f) d_hn += gl * head_w[v * d_model + d];
        }

        float cell_ln_out = wn_d * (h_new_d - cmean) * cinv + bn_d;
        sh_t_b[d] = cell_ln_out;
        __syncthreads();

        if (d == 0) {
            float m = 0;
            for (int i = 0; i < d_model; i++) m += sh_t_b[i];
            m /= (float)d_model;
            float v = 0;
            for (int i = 0; i < d_model; i++) { float df = sh_t_b[i] - m; v += df * df; }
            v /= (float)d_model;
            sh_t_d[0] = m;
            sh_t_d[1] = rsqrtf(v + 1e-5f);
        }
        __syncthreads();

        float hmean = sh_t_d[0];
        float hinv = sh_t_d[1];
        float h_norm_h = (cell_ln_out - hmean) * hinv;
        float hn_d = snw_d * h_norm_h + snb_d;

        for (int v = 0; v < vocab_size; v++) {
            float gl = grad_logits[boff_logits + v];
            if (gl != 0.0f) atomicAdd(&d_head_w[v * d_model + d], gl * hn_d);
        }
        if (d == 0) {
            for (int v = 0; v < vocab_size; v++) {
                float gl = grad_logits[boff_logits + v];
                if (gl != 0.0f) atomicAdd(&d_head_b[v], gl);
            }
        }

        float d_lno = d_hn * snw_d;
        sh_t_a[d] = d_lno;
        sh_t_c[d] = h_norm_h;
        __syncthreads();

        if (d == 0) {
            float sm = 0, ss = 0;
            for (int i = 0; i < d_model; i++) { sm += sh_t_a[i]; ss += sh_t_a[i] * sh_t_c[i]; }
            sh_t_d[0] = sm / (float)d_model;
            sh_t_d[1] = ss / (float)d_model;
        }
        __syncthreads();

        float m_h = sh_t_d[0];
        float s_h = sh_t_d[1];
        float d_state_head = (d_lno - m_h - h_norm_h * s_h) * hinv;
        float d_state = d_state_head + d_h_next;

        atomicAdd(&d_sn_w[d], d_hn * h_norm_h);
        atomicAdd(&d_sn_b[d], d_hn);

        float h_norm_c = (h_new_d - cmean) * cinv;
        float d_lno_c = d_state * wn_d;
        sh_t_a[d] = d_lno_c;
        sh_t_b[d] = h_norm_c;
        __syncthreads();

        if (d == 0) {
            float sm = 0, ss = 0;
            for (int i = 0; i < d_model; i++) { sm += sh_t_a[i]; ss += sh_t_a[i] * sh_t_b[i]; }
            sh_t_d[0] = sm / (float)d_model;
            sh_t_d[1] = ss / (float)d_model;
        }
        __syncthreads();

        float m_c = sh_t_d[0];
        float s_c = sh_t_d[1];
        float d_h_new_loc = (d_lno_c - m_c - h_norm_c * s_c) * cinv;

        atomicAdd(&d_wn[d], d_state * h_norm_c);
        atomicAdd(&d_bn[d], d_state);

        float d_h_ssm_total;
        float d_xp;
        float d_alpha_loc = 0.0f;
        float d_pa = 0.0f, d_pb = 0.0f, d_pal = 0.0f;

        if (do_full) {
            if (d < n_patterns) sh_d_attn[d] = saved_attn[t_off_np + d];
            __syncthreads();

            float attracted = 0;
            for (int p = 0; p < n_patterns; p++) {
                attracted += sh_d_attn[p] * patterns[p * d_model + d];
            }
            sh_t_c[d] = attracted;

            d_alpha_loc = d_h_new_loc * (attracted - h_ssm_d);
            float d_h_ssm_init = d_h_new_loc * (1.0f - alpha_d);
            float d_attracted_loc = d_h_new_loc * alpha_d;

            sh_t_a[d] = d_attracted_loc;
            sh_t_b[d] = d_h_ssm_init;
            __syncthreads();

            for (int p = d; p < n_patterns; p += d_model) {
                float val = 0;
                for (int k = 0; k < d_model; k++) {
                    val += sh_t_a[k] * patterns[p * d_model + k];
                }
                scratch_d_attn[b * n_patterns + p] = val;
            }
            __syncthreads();

            if (d == 0) {
                float dot_val = 0;
                for (int p = 0; p < n_patterns; p++) {
                    dot_val += sh_d_attn[p] * scratch_d_attn[b * n_patterns + p];
                }
                sh_t_d[0] = dot_val;
            }
            __syncthreads();

            float dot = sh_t_d[0];
            for (int p = d; p < n_patterns; p += d_model) {
                float da = scratch_d_attn[b * n_patterns + p];
                scratch_d_scores[b * n_patterns + p] = sh_d_attn[p] * (da - dot);
            }
            __syncthreads();

            float d_h_ssm_fs = 0;
            for (int p = 0; p < n_patterns; p++) {
                d_h_ssm_fs += scratch_d_scores[b * n_patterns + p] * beta * patterns[p * d_model + d];
            }

            d_h_ssm_total = sh_t_b[d] + d_h_ssm_fs;

            float dattr_d = sh_t_a[d];
            for (int p = 0; p < n_patterns; p++) {
                atomicAdd(&d_patterns[p * d_model + d], sh_d_attn[p] * dattr_d);
            }
        } else {
            d_h_ssm_total = d_h_new_loc;
        }
        __syncthreads();

        float h_prev_d;
        if (t > 0) {
            int t1_off = b * seq_len * d_model + (t - 1) * d_model;
            float hn1 = saved_h_new[t1_off + d];
            float cm1 = saved_cmean[b * seq_len + (t - 1)];
            float ci1 = saved_cinv_std[b * seq_len + (t - 1)];
            h_prev_d = wn_d * (hn1 - cm1) * ci1 + bn_d;
        } else {
            h_prev_d = h_init_fwd[b * d_model + d];
        }
        sh_h_prev[d] = h_prev_d;
        sh_emb_t[d] = emb_d;

        float xp_d = 0;
        for (int k = 0; k < d_model; k++) xp_d += emb[t_off_dm + k] * w_p[d * d_model + k];
        xp_d += b_p[d];
        sh_t_a[d] = xp_d;
        __syncthreads();

        float d_h_grad = d_h_ssm_total * gate_a_d;
        d_xp = d_h_ssm_total * gate_b_d;

        sh_t_b[d] = d_xp;
        sh_t_c[d] = d_h_grad;
        __syncthreads();

        float d_x_grad = 0;
        for (int k = 0; k < d_model; k++) {
            d_x_grad += sh_t_b[k] * w_p[k * d_model + d];
        }

        float dxp_local = sh_t_b[d];
        for (int k = 0; k < d_model; k++) {
            atomicAdd(&d_wp[d * d_model + k], dxp_local * sh_emb_t[k]);
        }
        atomicAdd(&d_bp[d], dxp_local);

        d_pa = d_h_ssm_total * h_prev_d * gate_a_d * (1.0f - gate_a_d);
        d_pb = d_h_ssm_total * xp_d * gate_b_d * (1.0f - gate_b_d);
        if (do_full) {
            d_pal = d_alpha_loc * alpha_d * (1.0f - alpha_d);
        }

        sh_t_a[d] = d_pa;
        sh_t_b[d] = d_pb;
        sh_t_c[d] = d_pal;
        __syncthreads();

        for (int k = 0; k < d_model; k++) {
            d_h_grad += sh_t_a[k] * w_a[k * d2 + d];
            d_x_grad += sh_t_a[k] * w_a[k * d2 + d_model + d];
            d_h_grad += sh_t_b[k] * w_b[k * d2 + d];
            d_x_grad += sh_t_b[k] * w_b[k * d2 + d_model + d];
            d_h_grad += sh_t_c[k] * w_g[k * d2 + d];
            d_x_grad += sh_t_c[k] * w_g[k * d2 + d_model + d];
        }

        float dpa_loc = sh_t_a[d];
        float dpb_loc = sh_t_b[d];
        float dpal_loc = sh_t_c[d];
        for (int k = 0; k < d_model; k++) {
            atomicAdd(&d_wa[d * d2 + k],           dpa_loc * sh_h_prev[k]);
            atomicAdd(&d_wa[d * d2 + d_model + k], dpa_loc * sh_emb_t[k]);
            atomicAdd(&d_wb[d * d2 + k],           dpb_loc * sh_h_prev[k]);
            atomicAdd(&d_wb[d * d2 + d_model + k], dpb_loc * sh_emb_t[k]);
            atomicAdd(&d_wg[d * d2 + k],           dpal_loc * sh_h_prev[k]);
            atomicAdd(&d_wg[d * d2 + d_model + k], dpal_loc * sh_emb_t[k]);
        }
        atomicAdd(&d_ba[d], dpa_loc);
        atomicAdd(&d_bb[d], dpb_loc);
        atomicAdd(&d_bg[d], dpal_loc);

        d_emb[t_off_dm + d] = d_x_grad;

        if (t == seq_len - 1) {
            int tid = x_tokens[b_t];
            atomicAdd(&d_slot_table[tid * d_model + d], d_h_grad);
        }

        d_h_next = d_h_grad;
        __syncthreads();
    }

    atomicAdd(&d_h_init[b * d_model + d], d_h_next);
}


__global__ void cann_sequence_kernel(
    const float* h_init, const float* emb, const int* x_tokens,
    const float* patterns, const float* slot_table,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    const float* head_w, const float* head_b,
    const float* sn_w, const float* sn_b,
    float* logits_out, int batch, int seq_len, int d_model,
    int n_patterns, int vocab_size, float beta, int attract_every
) {
    int b = blockIdx.x, d = threadIdx.x, d2 = d_model * 2;
    extern __shared__ float smem[];
    float* state = &smem[0];
    float* hbuf = &smem[d_model];
    float* sc = &smem[2 * d_model];
    if (d < d_model) state[d] = h_init[b * d_model + d];
    __syncthreads();
    for (int t = 0; t < seq_len; t++) {
        int do_full = (attract_every < 1) || (t % attract_every == attract_every - 1);
        const float* x = &emb[b * seq_len * d_model + t * d_model];
        if (t == seq_len - 1 && d < d_model) {
            int tid = x_tokens[b * seq_len + t];
            state[d] += slot_table[tid * d_model + d];
        }
        __syncthreads();
        if (d < d_model) {
            float a = 0, bb = 0, xp = 0;
            for (int k = 0; k < d_model; k++) { a += w_a[d * d2 + k] * state[k]; bb += w_b[d * d2 + k] * state[k]; }
            for (int k = 0; k < d_model; k++) { a += w_a[d * d2 + d_model + k] * x[k]; bb += w_b[d * d2 + d_model + k] * x[k]; xp += w_p[d * d_model + k] * x[k]; }
            a += b_a[d]; bb += b_b[d]; xp += b_p[d];
            float av = 1.0f / (1.0f + expf(-a)), bv = 1.0f / (1.0f + expf(-bb));
            hbuf[d] = av * state[d] + bv * xp;
        }
        __syncthreads();
        if (do_full) {
            if (d < n_patterns) {
                float s = 0;
                for (int k = 0; k < d_model; k++) s += patterns[d * d_model + k] * hbuf[k];
                sc[d] = s * beta;
            }
            __syncthreads();
            if (d == 0) {
                float mx = -FLT_MAX; for (int i = 0; i < n_patterns; i++) if (sc[i] > mx) mx = sc[i];
                float se = 0; for (int i = 0; i < n_patterns; i++) { sc[i] = expf(sc[i] - mx); se += sc[i]; }
                for (int i = 0; i < n_patterns; i++) sc[i] /= se;
            }
            __syncthreads();
        }
        if (d < d_model) {
            float hn;
            if (do_full) {
                float at = 0; for (int i = 0; i < n_patterns; i++) at += patterns[i * d_model + d] * sc[i];
                float al = 0; for (int k = 0; k < d_model; k++) al += w_g[d * d2 + k] * state[k];
                for (int k = 0; k < d_model; k++) al += w_g[d * d2 + d_model + k] * x[k];
                al += b_g[d]; float alv = 1.0f / (1.0f + expf(-al));
                hn = hbuf[d] + alv * (at - hbuf[d]);
            } else {
                hn = hbuf[d];
            }
            state[d] = hn;
        }
        __syncthreads();
        if (d == 0) {
            float m = 0; for (int i = 0; i < d_model; i++) m += state[i]; m /= (float)d_model;
            float v = 0; for (int i = 0; i < d_model; i++) { float df = state[i] - m; v += df * df; } v /= (float)d_model;
            float is = rsqrtf(v + 1e-5f);
            for (int i = 0; i < d_model; i++) state[i] = w_n[i] * (state[i] - m) * is + b_n[i];
        }
        __syncthreads();
        if (d == 0 && t == seq_len - 1) {
            float m = 0; for (int i = 0; i < d_model; i++) m += state[i]; m /= (float)d_model;
            float v = 0; for (int i = 0; i < d_model; i++) { float df = state[i] - m; v += df * df; } v /= (float)d_model;
            float is = rsqrtf(v + 1e-5f);
            for (int i = 0; i < d_model; i++) sc[i] = sn_w[i] * (state[i] - m) * is + sn_b[i];
        }
        __syncthreads();
        if (t == seq_len - 1 && d < vocab_size) {
            float lg = 0;
            for (int k = 0; k < d_model; k++) lg += head_w[d * d_model + k] * sc[k];
            lg += head_b[d];
            logits_out[b * seq_len * vocab_size + t * vocab_size + d] = lg;
        }
        __syncthreads();
    }
}


__declspec(dllexport) void launch_cann_step(
    const float* h_in, const float* x, const float* patterns,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    float* h_out, int d_model, int n_patterns, float beta
) {
    int nt = d_model > n_patterns ? d_model : n_patterns;
    int sm = (d_model + n_patterns) * (int)sizeof(float);
    cann_step_kernel<<<1, nt, sm>>>(h_in, x, patterns,
        w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n, h_out, d_model, n_patterns, beta);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) printf("CUDA error in launch_cann_step: %s\n", cudaGetErrorString(e));
}


__declspec(dllexport) void launch_cann_sequence_v2(
    const float* h_init, const float* emb, const int* x_tokens,
    const float* patterns, const float* slot_table,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n, const float* sn_w, const float* sn_b,
    const float* head_w, const float* head_b,
    float* logits_out,
    float* saved_h_ssm, float* saved_gate_a, float* saved_gate_b,
    float* saved_alpha, float* saved_attn, float* saved_h_new,
    float* saved_cmean, float* saved_cinv_std,
    float* saved_hmean, float* saved_hinv_std,
    int batch, int seq_len, int d_model, int n_patterns, int vocab_size,
    float beta, int attract_every
) {
    int nt = d_model > n_patterns ? d_model : n_patterns;
    if (nt < vocab_size) nt = vocab_size;
    int sm = (3 * d_model + n_patterns) * (int)sizeof(float);
    cann_sequence_kernel_v2<<<batch, nt, sm>>>(h_init, emb, x_tokens, patterns, slot_table, w_a, b_a,
        w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n, sn_w, sn_b, head_w, head_b, logits_out,
        saved_h_ssm, saved_gate_a, saved_gate_b, saved_alpha, saved_attn, saved_h_new,
        saved_cmean, saved_cinv_std, saved_hmean, saved_hinv_std,
        batch, seq_len, d_model, n_patterns, vocab_size, beta, attract_every);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) printf("CUDA error in launch_cann_sequence_v2: %s\n", cudaGetErrorString(e));
}


__declspec(dllexport) void launch_cann_sequence_backward(
    const float* grad_logits,
    const float* saved_h_ssm, const float* saved_gate_a,
    const float* saved_gate_b, const float* saved_alpha, const float* saved_attn,
    const float* saved_h_new, const float* saved_cmean, const float* saved_cinv_std,
    const float* saved_hmean, const float* saved_hinv_std,
    const float* emb, const int* x_tokens, const float* patterns,
    const float* slot_table,
    const float* h_init,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    const float* head_w, const float* head_b,
    const float* sn_w, const float* sn_b,
    float* d_h_init, float* d_emb, float* d_patterns,
    float* d_wa, float* d_ba, float* d_wb, float* d_bb,
    float* d_wg, float* d_bg, float* d_wp, float* d_bp,
    float* d_wn, float* d_bn,
    float* d_head_w, float* d_head_b,
    float* d_sn_w, float* d_sn_b,
    float* d_slot_table,
    float* scratch_d_attn, float* scratch_d_scores,
    int batch, int seq_len, int d_model, int n_patterns, int vocab_size,
    float beta, int attract_every
) {
    int sm = (6 * d_model + n_patterns) * (int)sizeof(float);
    cann_sequence_backward_kernel<<<batch, d_model, sm>>>(
        grad_logits, saved_h_ssm, saved_gate_a, saved_gate_b, saved_alpha, saved_attn, saved_h_new,
        saved_cmean, saved_cinv_std, saved_hmean, saved_hinv_std,
        emb, x_tokens, patterns, slot_table, h_init,
        w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n,
        head_w, head_b, sn_w, sn_b,
        d_h_init, d_emb, d_patterns,
        d_wa, d_ba, d_wb, d_bb, d_wg, d_bg, d_wp, d_bp,
        d_wn, d_bn, d_head_w, d_head_b, d_sn_w, d_sn_b,
        d_slot_table, scratch_d_attn, scratch_d_scores,
        batch, seq_len, d_model, n_patterns, vocab_size, beta, attract_every);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) printf("CUDA error in launch_cann_sequence_backward: %s\n", cudaGetErrorString(e));
}


__declspec(dllexport) void launch_cann_sequence(
    const float* h_init, const float* emb, const int* x_tokens,
    const float* patterns, const float* slot_table,
    const float* w_a, const float* b_a, const float* w_b, const float* b_b,
    const float* w_g, const float* b_g, const float* w_p, const float* b_p,
    const float* w_n, const float* b_n,
    const float* head_w, const float* head_b,
    const float* sn_w, const float* sn_b,
    float* logits_out, int batch, int seq_len, int d_model,
    int n_patterns, int vocab_size, float beta, int attract_every
) {
    int nt = d_model > n_patterns ? d_model : n_patterns;
    if (nt < vocab_size) nt = vocab_size;
    int sm = (2 * d_model + n_patterns) * (int)sizeof(float);
    cann_sequence_kernel<<<batch, nt, sm>>>(h_init, emb, x_tokens, patterns, slot_table,
        w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n,
        head_w, head_b, sn_w, sn_b, logits_out,
        batch, seq_len, d_model, n_patterns, vocab_size, beta, attract_every);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) printf("CUDA error in launch_cann_sequence: %s\n", cudaGetErrorString(e));
}

} // extern "C"
