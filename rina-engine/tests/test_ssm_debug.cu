#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*, const int*, float*, int, int, int, cudaStream_t);

void check_err(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fprintf(stderr,"  CUDA ERROR after %s: %s\n", msg, cudaGetErrorString(e));
}

int main() {
    fprintf(stderr,"Step 1: load model\n");
    ModelConfig cfg; TensorMap w;
    if (!load_model("/tmp/test2.rinn", cfg, w)) { fprintf(stderr,"load fail\n"); return 1; }
    fprintf(stderr,"  dim=%d n_layers=%d heads=%d d_c=%d ssm_steps=%d\n", 
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.d_c, cfg.ssm_steps);
    
fprintf(stderr,"Step 2: test direct layer creation\n");
    auto test_create = [&](const char* name) {
        fprintf(stderr,"  testing %s...\n", name);
        fflush(stderr);
        auto* p = create_layer_by_type(name);
        if (!p) { fprintf(stderr,"  %s: create FAILED\n", name); return; }
        bool ok = p->init(cfg, w, 0);
        fprintf(stderr,"  %s init: %d\n", name, (int)ok);
        fflush(stderr);
        int sv = p->saved_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
        fprintf(stderr,"  %s saved: %d\n", name, sv);
        fflush(stderr);
        int ws = p->workspace_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
        fprintf(stderr,"  %s ws: %d\n", name, ws);
        fflush(stderr);
        delete p;
    };
    test_create("gqa");
    test_create("inertia_wave_ssm");
    test_create("sparse_gather_fa");
    
    fprintf(stderr,"Step 3: build layers\n");
    auto layers = build_layers(cfg, w);
    fprintf(stderr,"  %zu layers\n", layers.size());
    if (layers.empty()) { fprintf(stderr,"  FAIL: no layers\n"); return 1; }

    fprintf(stderr,"Step 3: allocate buffers\n");
    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int hd = d*4*2/3/256*256;
    int ws = compute_workspace_per_token(cfg, layers);
    int total_saved = 0;
    for(auto& l : layers) total_saved += l->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    fprintf(stderr,"  n=%d d=%d hd=%d ws=%d V=%d saved=%d\n", n, d, hd, ws, V, total_saved);

    BufferManager bufs;
    bufs.alloc_fwd(n, d, ws, hd, V, total_saved);
    fprintf(stderr,"bufs: h=%p a=%p m=%p save=%p\n", bufs.fwd.h, bufs.fwd.a, bufs.fwd.m, bufs.fwd.save);

    // Allocate fm/fl
    cudaMalloc(&bufs.fwd.fm, n*sizeof(float));
    cudaMalloc(&bufs.fwd.fl, n*sizeof(float));

    cudaStream_t stream; cudaStreamCreate(&stream);
    int *ids; float *logits;
    cudaMalloc(&ids, n*sizeof(int));
    cudaMalloc(&logits, n*V*sizeof(float));
    
    float* h = bufs.fwd.h;
    auto* wte = w.get("transformer.wte.weight");
    if (!wte || !wte->data) { fprintf(stderr,"no wte\n"); return 1; }
    
    fprintf(stderr,"Embedding...\n");
    cudaGetLastError();
    launch_embedding_fp32((const float*)wte->data, ids, h, B, T, d, stream);
    cudaStreamSynchronize(stream);
    fprintf(stderr,"  %s\n", cudaGetErrorString(cudaGetLastError()));

    for (int l = 0; l < (int)layers.size(); l++) {
        fprintf(stderr,"\nLayer %d forward (type=%s)...\n", l, 
                l<(int)cfg.layers.size()?cfg.layers[l].type.c_str():"default");
        
        // Verify individual weight pointers
        auto check_w = [&](const char* name) {
            auto* t = w.get(name);
            fprintf(stderr,"  %s: %p\n", name, t && t->data ? t->data : nullptr);
        };
        if (l == 0) {
            check_w("transformer.h.0.ln1.weight");
            check_w("transformer.h.0.path.w_dq.weight");
            check_w("transformer.h.0.path.q_norm.weight");
            check_w("transformer.h.0.mlp.w1.weight");
        }

        float* layer_save = bufs.fwd.save + layers[l]->save_offset * n;
        bufs.fwd.save = layer_save;
        fprintf(stderr,"  save_ptr = %p (offset=%d)\n", layer_save, layers[l]->save_offset);
        
        cudaGetLastError();
        layers[l]->forward(h, bufs.fwd, B, T, stream);
        cudaStreamSynchronize(stream);
        cudaError_t e = cudaGetLastError();
        fprintf(stderr,"  result: %s\n", cudaGetErrorString(e));
        if (e != cudaSuccess) {
            const char* name = cudaGetErrorName(e);
            fprintf(stderr,"  error name: %s\n", name);
            break;
        }
    }

    w.free_all(); bufs.free_all();
    cudaFree(ids); cudaFree(logits); cudaStreamDestroy(stream);
    fprintf(stderr,"\nDone.\n");
    return 0;
}
