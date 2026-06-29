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

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"Usage: align_llama model.rinn \"id1 id2 ...\" [out.bin]\n"); return 1; }
    const char* model_path = argv[1];
    const char* ids_str = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "/tmp/align_logits.bin";

    std::vector<int> ids;
    const char* p = ids_str;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        ids.push_back(atoi(p));
        while (*p && *p != ' ') p++;
    }
    fprintf(stderr,"Input: %zu tokens\n", ids.size());

    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) { fprintf(stderr,"load fail\n"); return 1; }
    auto layers = build_layers(cfg, w);
    if (layers.empty()) { fprintf(stderr,"build fail\n"); return 1; }

    int B = 1, T = (int)ids.size(), n = B * T, d = cfg.dim, V = cfg.vocab_size;
    int ws = 0, total = 0;
    for (auto& l : layers) {
        int w = l->workspace_per_token(d, cfg.n_heads, cfg.head_dim);
        if (w > ws) ws = w;
        total += l->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    }
    BufferManager bufs;
    bufs.alloc_fwd(n, d, ws, 8192, V, total);
    if (!bufs.fwd.h) { fprintf(stderr,"alloc fail\n"); return 1; }

    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, n * sizeof(int));
    std::vector<int> h_ids(T);
    for (int i = 0; i < T; i++) h_ids[i] = i < (int)ids.size() ? ids[i] : 0;
    cudaMemcpy(d_ids, h_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    float* base_save = bufs.fwd.save;
    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte, d_ids, bufs.fwd.h, B, T, d, s);

    for (int l = 0; l < (int)layers.size(); l++) {
        bufs.fwd.save = base_save + layers[l]->save_offset * n;
        layers[l]->forward(bufs.fwd.h, bufs.fwd, B, T, s);
    }
    bufs.fwd.save = base_save;
    cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr,"forward: %s\n", cudaGetErrorString(e)); return 1; }
    fprintf(stderr,"forward OK\n");

    // LM head (inference uses LayerNorm for final norm; Llama uses RMSNorm)
    auto* ln_f = w.get("transformer.ln_f.weight");
    if (ln_f) {
        extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
        launch_rms_norm_fp32(bufs.fwd.h, (const float*)ln_f->data, n, d, 1e-5f, s);
    }

    const float* lm_w = (const float*)w.get("lm_head.weight")->data;
    launch_linear_fp32(bufs.fwd.h, lm_w, bufs.fwd.lm, n, V, d, s);
    cudaStreamSynchronize(s);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr,"lm_head fail\n"); return 1; }

    std::vector<float> cpu(n * V);
    cudaMemcpy(cpu.data(), bufs.fwd.lm, n * V * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(out_path, "wb");
    fwrite(cpu.data(), sizeof(float), n * V, f);
    fclose(f);
    fprintf(stderr,"Written %d logits -> %s\n", n * V, out_path);

    w.free_all(); bufs.free_all(); cudaFree(d_ids); cudaStreamDestroy(s);
    return 0;
}
