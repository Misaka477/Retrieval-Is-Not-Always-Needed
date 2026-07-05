// Single-layer bf16 vs fp32 comparison for debugging
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);

extern "C" {
static bool single_init(void* self, const ModelConfig& c, const TensorMap& w, int l) { return true; }
static void single_fwd(void*, float*, ForwardBuffers&, int, int, cudaStream_t) {}
static void single_bwd(void*, GradBuffers&, ForwardBuffers&, float*, int, int, cudaStream_t) {}
static int single_ws(void*, int, int, int) { return 0; }
static int single_sv(void*, int, int, int) { return 0; }
static void single_dst(void*) {}
}
static const LayerVTable passthru_vtab = { single_init, single_fwd, single_bwd, single_ws, single_sv, single_dst };

struct IdentityLayer {
    Layer l;
    IdentityLayer() { l.impl = nullptr; l.vtab = &passthru_vtab; }
};

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"Usage: test_bf16_single model.rinn \"ids\"\n"); return 1; }
    const char* model_path = argv[1];
    const char* ids_str = argv[2];

    std::vector<int> ids;
    const char* p = ids_str; while(*p) { while(*p==' ')p++; if(!*p)break; ids.push_back(atoi(p)); while(*p&&*p!=' ')p++; }
    fprintf(stderr,"Tokens: %zu\n",ids.size());

    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) { fprintf(stderr,"load fail\n"); return 1; }

    int B=1, T=(int)ids.size(), n=B*T, d=cfg.dim, V=cfg.vocab_size;

    // Build one fp32 layer and one bf16 layer
    auto fp32_l = std::unique_ptr<Layer>(create_layer_by_type("gqa"));
    auto bf16_l = std::unique_ptr<Layer>(create_layer_by_type("gqa_bf16"));
    if (!fp32_l || !fp32_l->init(cfg, w, 0)) { fprintf(stderr,"fp32 init fail\n"); return 1; }
    if (!bf16_l || !bf16_l->init(cfg, w, 0)) { fprintf(stderr,"bf16 init fail\n"); return 1; }

    int ws  = fp32_l->workspace_per_token(d, cfg.n_heads, cfg.head_dim);
    int sp  = fp32_l->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    fprintf(stderr,"d=%d H=%d Hkv=%d dh=%d hd=%d ws=%d sp=%d\n", d, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, d*4*2/3/256*256, ws, sp);

    // Buffer for layer 0 output comparison
    BufferManager bufs[2];
    bufs[0].alloc_fwd(n, d, ws, 8192, V, sp);
    bufs[1].alloc_fwd(n, d, ws, 8192, V, sp);

    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, n*sizeof(int));
    cudaMemcpy(d_ids, ids.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    const float* wte = (const float*)w.get("transformer.wte.weight")->data;

    // Compare layer by layer (up to 3 layers for debugging)
    int n_layers_to_test = std::min(3, cfg.n_layers);
    for (int layer_to_test = 0; layer_to_test < n_layers_to_test; layer_to_test++) {
        // Re-init layers for this specific layer index
        fp32_l->init(cfg, w, layer_to_test);
        bf16_l->init(cfg, w, layer_to_test);

        // Reset: re-embed
        launch_embedding_fp32(wte, d_ids, bufs[0].fwd.h, B, T, d, s);
        launch_embedding_fp32(wte, d_ids, bufs[1].fwd.h, B, T, d, s);

        // Run all layers up to and including layer_to_test
        for (int l = 0; l <= layer_to_test; l++) {
            auto f32l = std::unique_ptr<Layer>(create_layer_by_type("gqa"));
            auto b16l = std::unique_ptr<Layer>(create_layer_by_type("gqa_bf16"));
            f32l->init(cfg, w, l);
            b16l->init(cfg, w, l);
            f32l->forward(bufs[0].fwd.h, bufs[0].fwd, B, T, s);
            b16l->forward(bufs[1].fwd.h, bufs[1].fwd, B, T, s);
        }
        cudaStreamSynchronize(s);

        std::vector<float> h0(n*d), h1(n*d);
        cudaMemcpy(h0.data(), bufs[0].fwd.h, n*d*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h1.data(), bufs[1].fwd.h, n*d*sizeof(float), cudaMemcpyDeviceToHost);

        double max_diff=0, sum_sq=0, fp32_norm=0;
        int worst_i=0;
        for(int i=0;i<n*d;i++){
            double diff=fabs(h0[i]-h1[i]);
            if(diff>max_diff){max_diff=diff;worst_i=i;}
            sum_sq+=diff*diff; fp32_norm+=h0[i]*h0[i];
        }
        double rmse=sqrt(sum_sq/(n*d));
        double snr=20*log10(sqrt(fp32_norm/(n*d))/(rmse+1e-30));
        fprintf(stderr,"Layer %d: max_diff=%.2e RMSE=%.2e SNR=%.1f dB worst=(%d: %.4f vs %.4f)\n",
            layer_to_test, max_diff, rmse, snr, worst_i, h0[worst_i], h1[worst_i]);
    }

    w.free_all(); bufs[0].free_all(); bufs[1].free_all();
    cudaFree(d_ids); cudaStreamDestroy(s);
    return 0;
}
