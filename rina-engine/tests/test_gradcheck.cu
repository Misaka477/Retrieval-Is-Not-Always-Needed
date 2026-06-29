#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "kernels/gemm.cuh"
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"

// ── Kernel declarations ──
extern void launch_silu_fp32(float*, int, cudaStream_t);
extern void launch_silu_bwd_fp32(const float*, const float*, float*, int, cudaStream_t);
extern void launch_silu_mul_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, cudaStream_t);
extern void launch_sigmoid_bwd_fp32(float*, const float*, int, cudaStream_t);
extern void launch_linear_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, int, cudaStream_t);
extern float launch_crossentropy_fp32(const float*, const int*, float*, int, int, cudaStream_t);
extern void launch_embedding_bwd_fp32(const float*, const int*, float*, int, int, int, cudaStream_t);
extern void launch_rope_fp32(float*, const float*, const float*, int, int, int, int, cudaStream_t);
extern void launch_rope_bwd_fp32(float*, const float*, const float*, const float*,
    int, int, int, int, cudaStream_t);
extern void launch_ssm_scan_bwd_fp32(const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int, cudaStream_t);
extern void launch_ssm_scan_inline(const float*, const float*, float*, int, int, int, int, cudaStream_t);
extern void launch_silu_mul_inline(float*, const float*, const float*, int, cudaStream_t);
extern void launch_flashattn_fwd_save_stats(const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_flash_attn_bwd_fp32(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t);

static const float MAX_ERR = 2e-4f;
static int total = 0, passed = 0;

#define REF_DIR "/tmp/gradcheck_ref"

// ── Load .bin + shape file from reference ──
static std::vector<float> load_bin(const std::string& name, std::vector<int>& shape) {
    std::string spath = std::string(REF_DIR) + "/" + name + "_shape.txt";
    FILE* sf = fopen(spath.c_str(), "r");
    if (!sf) { fprintf(stderr, "  can't open %s\n", spath.c_str()); return {}; }
    shape.clear();
    int s; while (fscanf(sf, "%d", &s) == 1) shape.push_back(s);
    fclose(sf);
    int n = 1; for (int si : shape) n *= si;
    
    std::string bpath = std::string(REF_DIR) + "/" + name + ".bin";
    FILE* bf = fopen(bpath.c_str(), "rb");
    if (!bf) { fprintf(stderr, "  can't open %s\n", bpath.c_str()); return {}; }
    std::vector<float> data(n);
    fread(data.data(), sizeof(float), n, bf);
    fclose(bf);
    return data;
}

static int nelems(const std::vector<int>& s) {
    int n = 1; for (int si : s) n *= si; return n;
}

// ── Upload to GPU ──
static float* to_gpu(const std::vector<float>& data) {
    float* d; cudaMalloc(&d, data.size() * sizeof(float));
    cudaMemcpy(d, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

static int* to_gpu_i(const std::vector<int>& data) {
    int* d; cudaMalloc(&d, data.size() * sizeof(int));
    cudaMemcpy(d, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// ── Compare ──
static bool check(const std::string& name, float* d_got, int n, const std::vector<float>& ref) {
    total++;
    std::vector<float> got(n);
    cudaMemcpy(got.data(), d_got, n * sizeof(float), cudaMemcpyDeviceToHost);
    float max_diff = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabs(ref[i] - got[i]);
        if (d > max_diff) max_diff = d;
        float rel = d / (fabs(ref[i]) + 1e-8f);
        if (rel > max_rel) max_rel = rel;
    }
    bool ok = max_diff < MAX_ERR;
    if (!ok)
        fprintf(stderr, "  FAIL %s: max_diff=%.2e max_rel=%.2e (limit=%.0e)\n",
            name.c_str(), max_diff, max_rel, MAX_ERR);
    else { printf("  OK   %s: max_diff=%.2e\n", name.c_str(), max_diff); passed++; }
    return ok;
}

static void check_free(float* p) { cudaFree(p); }

int main() {
    setbuf(stdout, NULL); setbuf(stderr, NULL);
    cudaStream_t stream; cudaStreamCreate(&stream);
    printf("=== Gradient Check Tests ===\n"); fflush(stdout);

    // 1. SiLU
    printf("\n─── SiLU ───\n"); fflush(stdout);
    {
        std::vector<int> sh; auto x = load_bin("silu_x", sh);
        int n = nelems(sh);
        auto dx_ref = load_bin("silu_dx", sh);
        float *d_x = to_gpu(x), *d_dx = to_gpu(dx_ref);
        // Run our backward: dout=1, gate=x
        std::vector<float> dout_v(n, 1.0f); float* d_dout = to_gpu(dout_v);
        float* d_got; cudaMalloc(&d_got, n * sizeof(float)); cudaMemset(d_got, 0, n * sizeof(float));
        launch_silu_bwd_fp32(d_dout, d_x, d_got, n, stream);
        cudaStreamSynchronize(stream);
        check("silu_bwd", d_got, n, dx_ref);
        check_free(d_x); check_free(d_dx); check_free(d_dout); check_free(d_got);
    }

    // 2. SiLU_MUL
    printf("\n─── SiLU_MUL ───\n");
    {
        std::vector<int> sh; auto gate = load_bin("silu_mul_gate", sh);
        int n = nelems(sh);
        auto up = load_bin("silu_mul_up", sh);
        auto dgate_ref = load_bin("silu_mul_dgate", sh);
        auto dup_ref = load_bin("silu_mul_dup", sh);
        float *d_gate = to_gpu(gate), *d_up = to_gpu(up);
        float *d_dgate, *d_dup; cudaMalloc(&d_dgate, n*4); cudaMalloc(&d_dup, n*4);
        cudaMemset(d_dgate, 0, n*4); cudaMemset(d_dup, 0, n*4);
        // dout = all ones
        std::vector<float> dout_v(n, 1.0f); float* d_dout = to_gpu(dout_v);
        launch_silu_mul_bwd_fp32(d_dout, d_gate, d_up, d_dgate, d_dup, n, stream);
        cudaStreamSynchronize(stream);
        check("silu_mul_dgate", d_dgate, n, dgate_ref);
        check("silu_mul_dup", d_dup, n, dup_ref);
        check_free(d_gate); check_free(d_up); check_free(d_dgate); check_free(d_dup); check_free(d_dout);
    }

    // 3. Linear
    printf("\n─── Linear ───\n");
    {
        std::vector<int> sx, sw;
        auto x = load_bin("linear_x", sx); auto w = load_bin("linear_w", sw);
        auto dx_ref = load_bin("linear_dx", sx); auto dw_ref = load_bin("linear_dw", sw);
        int M = sx[0], K = sx[1], N = sw[0];
        float *d_x = to_gpu(x), *d_w = to_gpu(w);
        std::vector<float> dout_v(M*N, 1.0f); float* d_dout = to_gpu(dout_v);
        float *d_dx, *d_dw; cudaMalloc(&d_dx, M*K*4); cudaMalloc(&d_dw, N*K*4);
        cudaMemset(d_dx, 0, M*K*4); cudaMemset(d_dw, 0, N*K*4);
        launch_linear_bwd_fp32(d_dout, d_x, d_w, d_dx, d_dw, M, N, K, stream);
        cudaStreamSynchronize(stream);
        check("linear_dx", d_dx, M*K, dx_ref);
        check("linear_dw", d_dw, N*K, dw_ref);
        check_free(d_x); check_free(d_w); check_free(d_dout); check_free(d_dx); check_free(d_dw);
    }

    // 4. LayerNorm - need both forward (to compute x_norm) and backward
    printf("\n─── LayerNorm ───\n");
    {
        std::vector<int> sx, sw;
        auto x = load_bin("layernorm_x", sx); auto w = load_bin("layernorm_w", sw);
        auto dx_ref = load_bin("layernorm_dx", sx);
        int N = sx[0], D = sx[1];
        float *d_x = to_gpu(x), *d_w = to_gpu(w);
        // Run LN forward (in-place)
        launch_pytorch_ln_kernel(d_x, d_w, N, D, 1e-5f, stream);
        cudaStreamSynchronize(stream);
        // Backward with saved x (but we used x as original, now it's overwritten)
        // Our backward needs the original x (before LN). We saved x_save for each call.
        // For the test, we need to re-upload x
        cudaMemcpy(d_x, x.data(), N*D*4, cudaMemcpyHostToDevice);
        float *d_dx; cudaMalloc(&d_dx, N*D*4); cudaMemset(d_dx, 0, N*D*4);
        std::vector<float> dout_v(N*D, 1.0f); float* d_dout = to_gpu(dout_v);
        launch_layernorm_bwd_fp32(d_dout, d_x, d_w, d_dx, nullptr, N, D, stream);
        cudaStreamSynchronize(stream);
        check("layernorm_dx", d_dx, N*D, dx_ref);
        check_free(d_x); check_free(d_w); check_free(d_dx); check_free(d_dout);
    }

    // 5. CrossEntropy
    printf("\n─── CrossEntropy ───\n");
    {
        std::vector<int> sh; auto logits = load_bin("crossentropy_logits", sh);
        std::vector<int> sh_t; auto targets_f = load_bin("crossentropy_targets", sh_t);
        auto dlogits_ref = load_bin("crossentropy_dlogits", sh);
        int N = sh[0], V = sh[1];
        // targets: int32, convert from float to int
        std::vector<int> targets(N);
        for(int i=0;i<N;i++) targets[i] = (int)targets_f[i];
        
        float *d_logits = to_gpu(logits), *d_dlogits;
        cudaMalloc(&d_dlogits, N*V*4); cudaMemset(d_dlogits, 0, N*V*4);
        int* d_tgt = to_gpu_i(targets);
        launch_crossentropy_fp32(d_logits, d_tgt, d_dlogits, N, V, stream);
        cudaStreamSynchronize(stream);
        check("crossentropy_dlogits", d_dlogits, N*V, dlogits_ref);
        check_free(d_logits); check_free(d_dlogits); cudaFree(d_tgt);
    }

    // 6. Sigmoid
    printf("\n─── Sigmoid ───\n");
    {
        std::vector<int> sh; auto x = load_bin("sigmoid_x", sh);
        auto dx_ref = load_bin("sigmoid_dx", sh);
        int n = nelems(sh);
        float *d_x = to_gpu(x);
        // Our backward: dout *= sigmoid'(x)
        std::vector<float> dout_v(n, 1.0f); float* d_dout = to_gpu(dout_v);
        launch_sigmoid_bwd_fp32(d_dout, d_x, n, stream);
        cudaStreamSynchronize(stream);
        check("sigmoid_bwd", d_dout, n, dx_ref);
        check_free(d_x); check_free(d_dout);
    }

    // 7. Embedding
    printf("\n─── Embedding ───\n");
    {
        std::vector<int> sh_w, sh_idx;
        auto weight = load_bin("embedding_weight", sh_w);
        auto idx_f = load_bin("embedding_idx", sh_idx);
        auto dw_ref = load_bin("embedding_d_weight", sh_w);
        int B = sh_idx[0], T = sh_idx[1], D = sh_w[1];
        std::vector<int> idx(B*T);
        for(int i=0;i<B*T;i++) idx[i] = (int)idx_f[i];
        
        // dout: ones at first token positions for each embedding index
        std::vector<float> dout(B*T*D, 1.0f);
        float *d_dout = to_gpu(dout), *d_dw;
        cudaMalloc(&d_dw, sh_w[0]*D*4); cudaMemset(d_dw, 0, sh_w[0]*D*4);
        int* d_idx = to_gpu_i(idx);
        launch_embedding_bwd_fp32(d_dout, d_idx, d_dw, B, T, D, stream);
        cudaStreamSynchronize(stream);
        check("embedding_dw", d_dw, sh_w[0]*D, dw_ref);
        check_free(d_dout); check_free(d_dw); cudaFree(d_idx);
    }

    // 8. RoPE
    printf("\n─── RoPE ───\n");
    {
        std::vector<int> sh; auto x = load_bin("rope_x", sh);
        auto dx_ref = load_bin("rope_dx", sh);
        std::vector<int> sh_c; auto cos = load_bin("rope_cos", sh_c);
        std::vector<int> sh_s; auto sin = load_bin("rope_sin", sh_s);
        int n = sh[0], d = sh[1], T = sh_c[0];
        int B = 2, TT = T, H = 4, d_ = d;
        
        int nelem = n * d_;
        float *d_cos = to_gpu(cos), *d_sin = to_gpu(sin);
        float *d_out; cudaMalloc(&d_out, nelem*4);
        std::vector<float> ones(nelem, 1.0f); cudaMemcpy(d_out, ones.data(), nelem*4, cudaMemcpyHostToDevice);
        float *d_dx; cudaMalloc(&d_dx, nelem*4); cudaMemset(d_dx, 0, nelem*4);
        launch_rope_bwd_fp32(d_dx, d_out, d_cos, d_sin, B, TT, H, d_, stream);
        cudaStreamSynchronize(stream);
        check("rope_dx", d_dx, nelem, dx_ref);
        cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_out); cudaFree(d_dx);
    }

    // 9. SSM Scan
    printf("\n─── SSM Scan ───\n"); fflush(stdout);
    {
        std::vector<int> sh_m; auto mem = load_bin("ssm_scan_mem", sh_m);
        std::vector<int> sh_d; auto decay = load_bin("ssm_scan_decay", sh_d);
        auto dmem_ref = load_bin("ssm_scan_dmem", sh_m);
        auto ddecay_ref = load_bin("ssm_scan_ddecay", sh_d);
        int B=1, T=8, H=2, dh=4;
        int nm = B*T*H*dh, nd = B*T*H;
        
        float *d_mem = to_gpu(mem), *d_dec = to_gpu(decay);
        float *d_sf; cudaMalloc(&d_sf, nm*4);
        float *d_dmem; cudaMalloc(&d_dmem, nm*4); cudaMemset(d_dmem, 0, nm*4);
        float *d_ddec; cudaMalloc(&d_ddec, nd*4); cudaMemset(d_ddec, 0, nd*4);
        
        // Forward
        launch_ssm_scan_inline(d_mem, d_dec, d_sf, B, T, H, dh, stream);
        cudaStreamSynchronize(stream);
        // Backward (dout = all ones)
        std::vector<float> dout_v(nm, 1.0f); float* d_dout = to_gpu(dout_v);
        launch_ssm_scan_bwd_fp32(d_dout, d_mem, d_dec, d_sf, d_dmem, d_ddec, B, T, H, dh, stream);
        cudaStreamSynchronize(stream);
        check("ssm_dmem", d_dmem, nm, dmem_ref);
        check("ssm_ddecay", d_ddec, nd, ddecay_ref);
        cudaFree(d_mem); cudaFree(d_dec); cudaFree(d_sf); cudaFree(d_dmem); cudaFree(d_ddec); cudaFree(d_dout);
    }

    // 10. FlashAttention
    printf("\n─── FlashAttention ───\n"); fflush(stdout);
    {
        int B=1, H=4, T=8, dq=48, dh=32, dhr=16;
        int Bh = B*H;  // Actually head_dim=32, d_c isn't relevant here
        // Use simple shapes for FA test: B=1, H=2, T=8, dq=16, dh=16
        int B_=1, H_=2, T_=8, dq_=16, dh_=16;
        int nq = B_*H_*T_*dq_, nk = nq, nv = B_*H_*T_*dh_;
        
        // Generate random Q, K, V
        std::mt19937 rng(42);
        auto rand_vec = [&](int n) {
            std::vector<float> v(n);
            for (int i=0; i<n; i++) v[i] = (float)rng()/rng.max() * 2 - 1;
            return v;
        };
        auto Qh = rand_vec(nq), Kh = rand_vec(nq), Vh = rand_vec(nv);
        
        float *d_Q = to_gpu(Qh), *d_K = to_gpu(Kh), *d_V = to_gpu(Vh);
        float *d_O; cudaMalloc(&d_O, nv*4);
        float *d_m, *d_l; cudaMalloc(&d_m, B_*H_*T_*4); cudaMalloc(&d_l, B_*H_*T_*4);
        
        // Forward with save stats
        launch_flashattn_fwd_save_stats(d_Q, d_K, d_V, d_O, d_m, d_l, B_, H_, T_, dq_, dh_, stream);
        cudaStreamSynchronize(stream);
        
        // Backward: dout = all ones at O
        float *d_dO = to_gpu(std::vector<float>(nv, 1.0f));
        float *d_dQ, *d_dK, *d_dV;
        cudaMalloc(&d_dQ, nq*4); cudaMalloc(&d_dK, nq*4); cudaMalloc(&d_dV, nv*4);
        cudaMemset(d_dQ, 0, nq*4); cudaMemset(d_dK, 0, nq*4); cudaMemset(d_dV, 0, nv*4);
        
        launch_flash_attn_bwd_fp32(d_Q, d_K, d_V, d_O, d_dO, d_m, d_l,
            d_dQ, d_dK, d_dV, B_, H_, T_, dq_, dh_, stream);
        cudaStreamSynchronize(stream);
        
        // Verify: gradient norm should be finite (no NaN)
        std::vector<float> got_dq(nq), got_dk(nq), got_dv(nv);
        cudaMemcpy(got_dq.data(), d_dQ, nq*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_dk.data(), d_dK, nq*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_dv.data(), d_dV, nv*4, cudaMemcpyDeviceToHost);
        float max_dq = 0, max_dk = 0, max_dv = 0;
        for (float v : got_dq) { float a = fabs(v); if (a > max_dq) max_dq = a; }
        for (float v : got_dk) { float a = fabs(v); if (a > max_dk) max_dk = a; }
        for (float v : got_dv) { float a = fabs(v); if (a > max_dv) max_dv = a; }
        bool fa_ok = std::isfinite(max_dq) && std::isfinite(max_dk) && std::isfinite(max_dv);
        if (!fa_ok) fprintf(stderr, "  FAIL flash_attn_bwd: NaN detected\n");
        else printf("  OK   flash_attn_bwd: |dQ|max=%.2e |dK|max=%.2e |dV|max=%.2e\n", max_dq, max_dk, max_dv);
        total++; if (fa_ok) passed++;
        
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_m); cudaFree(d_l); cudaFree(d_dO);
        cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    }

    cudaStreamDestroy(stream);
    printf("\n─── Results: %d/%d passed ───\n", passed, total);
    return passed == total ? 0 : 1;
}
