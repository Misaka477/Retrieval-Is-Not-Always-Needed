#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

static void dump_buf(const char* label, const float* buf, int n) {
    char path[256]; snprintf(path,256,"/tmp/%s.bin",label);
    std::vector<float> cpu(n);
    cudaMemcpy(cpu.data(), buf, n*sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb"); fwrite(cpu.data(), sizeof(float), n, f); fclose(f);
    fprintf(stderr,"  dumped %s (%d floats)\n", label, n);
}

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern void launch_rope_fp32(float*,const float*,const float*,int,int,int,int,cudaStream_t);
extern void build_qkv_fp32_kernel(const float*,const float*,const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,int,int,cudaStream_t);
extern void launch_flash_attn_fp32(const float*,const float*,const float*,float*,int,int,int,int,int,cudaStream_t);
extern void launch_transpose_attn(float*,const float*,int,int,int,cudaStream_t);

int main() {
    ModelConfig cfg; TensorMap w;
    if (!load_model("/tmp/llama3.2-1b.rinn", cfg, w)) { fprintf(stderr,"load fail\n"); return 1; }
    auto layers = build_layers(cfg, w);
    if (layers.empty()) return 1;

    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int H=cfg.n_heads,Hkv=cfg.n_kv_heads,dh=cfg.head_dim;
    int hq = n*H*dh, hk = n*Hkv*dh, hdh = H*dh;
    int ws=0,total=0;
    for (auto& l : layers) {
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim); if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,8192,V,total);
    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, n*sizeof(int));
    int ids[] = {8270,1860,6390,6191,6734,7265,1466,5426};
    cudaMemcpy(d_ids, ids, n*sizeof(int), cudaMemcpyHostToDevice);
    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    float* h = bufs.fwd.h;

    // Embedding
    launch_embedding_fp32(wte, d_ids, h, B, T, d, s);
    cudaStreamSynchronize(s); dump_buf("embed", h, n*d);

    // Layer 0 - step by step
    float* a = bufs.fwd.a;
    float* m = bufs.fwd.m;

    // LN1 (RMSNorm)
    auto* ln1_w = (const float*)w.get("transformer.h.0.ln1.weight")->data;
    cudaMemcpyAsync(a, h, n*d*sizeof(float), cudaMemcpyDeviceToDevice, s);
    launch_rms_norm_fp32(a, ln1_w, n, d, 1e-5f, s);
    cudaStreamSynchronize(s); dump_buf("ln1_out", a, n*d);

    // Q proj
    auto* wq = (const float*)w.get("transformer.h.0.attn.w_q.weight")->data;
    launch_linear_fp32(a, wq, m, n, hdh, d, s);
    cudaStreamSynchronize(s); dump_buf("q_proj", m, hq);

    // K proj
    auto* wk = (const float*)w.get("transformer.h.0.attn.w_k.weight")->data;
    launch_linear_fp32(a, wk, m + hq, n, hk, d, s);
    cudaStreamSynchronize(s); dump_buf("k_proj", m + hq, hk);

    // V proj
    auto* wv = (const float*)w.get("transformer.h.0.attn.w_v.weight")->data;
    launch_linear_fp32(a, wv, m + hq + hk, n, hk, d, s);
    cudaStreamSynchronize(s); dump_buf("v_proj", m + hq + hk, hk);

    // RoPE on Q
    auto* rqc = (const float*)w.get("transformer.h.0.attn.rope_q.cos")->data;
    auto* rqs = (const float*)w.get("transformer.h.0.attn.rope_q.sin")->data;
    launch_rope_fp32(m, rqc, rqs, B, T, H, dh, s);
    // RoPE on K
    auto* rkc = (const float*)w.get("transformer.h.0.attn.rope.cos")->data;
    auto* rks = (const float*)w.get("transformer.h.0.attn.rope.sin")->data;
    launch_rope_fp32(m + hq, rkc, rks, B, T, Hkv, dh, s);
    cudaStreamSynchronize(s); dump_buf("q_rope", m, hq);
    dump_buf("k_rope", m + hq, hk);

    // Build Qf/Kf/Vf for FlashAttention
    float* Qf = m + hq + hk * 2;
    float* Kf = Qf + n * H * dh;
    float* Vf = Kf + n * H * dh;
    build_qkv_fp32_kernel(m, m+hq, m+hq+hk, m, m+hq, Qf, Kf, Vf, B, T, H, Hkv, dh, 0, dh, s);
    cudaStreamSynchronize(s);
    launch_flash_attn_fp32(Qf, Kf, Vf, Qf, B, H, T, dh, dh, s);
    cudaStreamSynchronize(s); dump_buf("flash_out", Qf, n*hdh);

    // Transpose + O proj
    launch_transpose_attn(a, Qf, H, T, dh, s);
    auto* wo = (const float*)w.get("transformer.h.0.attn.w_o.weight")->data;
    launch_linear_fp32(a, wo, a, n, d, hdh, s);
    cudaStreamSynchronize(s); dump_buf("attn_out", a, n*d);

    fprintf(stderr,"\nDone\n");
    w.free_all(); bufs.free_all(); cudaFree(d_ids); cudaStreamDestroy(s);
    return 0;
}
