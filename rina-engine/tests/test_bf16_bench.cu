// bf16 vs fp32 speed benchmark
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"Usage: test_bf16_bench model.rinn \"ids\" [iters]\n"); return 1; }
    const char* model_path = argv[1];
    const char* ids_str = argv[2];
    int iters = (argc > 3) ? atoi(argv[3]) : 10;

    std::vector<int> ids;
    const char* p = ids_str; while(*p) { while(*p==' ')p++; if(!*p)break; ids.push_back(atoi(p)); while(*p&&*p!=' ')p++; }
    fprintf(stderr,"Tokens: %zu, iters: %d\n", ids.size(), iters);

    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) { fprintf(stderr,"load fail\n"); return 1; }

    int B=1, T=(int)ids.size(), n=B*T, d=cfg.dim, V=cfg.vocab_size;

    auto fp32_layers = build_layers(cfg, w);
    // Create bf16 layers
    auto bf16_layers = std::vector<std::unique_ptr<Layer>>();
    for (int i = 0; i < cfg.n_layers; i++) {
        std::string type = "gqa_bf16";
        auto l = std::unique_ptr<Layer>(create_layer_by_type(type));
        if (!l || !l->init(cfg, w, i)) {
            fprintf(stderr,"bf16 layer %d init fail\n", i); return 1;
        }
        bf16_layers.push_back(std::move(l));
    }

    int ws=0,sp=0;
    for(auto& l:fp32_layers){int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;sp+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);}

    BufferManager bf32, bf16;
    bf32.alloc_fwd(n, d, ws, 8192, V, sp);
    bf16.alloc_fwd(n, d, ws, 8192, V, sp);

    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, n*sizeof(int));
    cudaMemcpy(d_ids, ids.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    auto run_forward = [&](auto& layers, auto& bufs) {
        const float* wte = (const float*)w.get("transformer.wte.weight")->data;
        launch_embedding_fp32(wte, d_ids, bufs.fwd.h, B, T, d, s);
        float* base_save = bufs.fwd.save;
        for (int l = 0; l < (int)layers.size(); l++) {
            bufs.fwd.save = base_save + layers[l]->save_offset * n;
            layers[l]->forward(bufs.fwd.h, bufs.fwd, B, T, s);
        }
        bufs.fwd.save = base_save;
        cudaStreamSynchronize(s);
    };

    // Warmup
    run_forward(fp32_layers, bf32);
    run_forward(bf16_layers, bf16);

    // Benchmark fp32
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_forward(fp32_layers, bf32);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;

    // Benchmark bf16
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_forward(bf16_layers, bf16);
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;

    fprintf(stderr,"\n=== Speed benchmark (T=%d, %d iters) ===\n", T, iters);
    fprintf(stderr,"  fp32: %.2f ms\n", fp32_ms);
    fprintf(stderr,"  bf16: %.2f ms\n", bf16_ms);
    fprintf(stderr,"  Speedup: %.2fx\n", fp32_ms / bf16_ms);

    w.free_all(); bf32.free_all(); bf16.free_all();
    cudaFree(d_ids); cudaStreamDestroy(s);
    return 0;
}
