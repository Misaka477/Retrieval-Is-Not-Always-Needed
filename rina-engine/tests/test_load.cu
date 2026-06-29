#include <cstdio>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"
int main(int argc, char** argv) {
    ModelConfig cfg; TensorMap w;
    if (!load_model(argv[1], cfg, w)) { fprintf(stderr, "Load failed\n"); return 1; }
    fprintf(stderr, "Model: %s dim=%d L=%d H=%d/%d V=%d dc=%d\n",
        cfg.name.c_str(), cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.vocab_size, cfg.d_c);
    int c=0;
    for (auto& [n, wt] : w.tensors) {
        c++;
        if (wt.quant_type == QuantType::FP32) {
            fprintf(stderr, "  %s [%d,%d] n=%d\n", n.c_str(), wt.shape[0], wt.shape[1], wt.n_elems);
        }
    }
    fprintf(stderr, "Total: %d tensors\n", c);

    int B=1,T=2,V=cfg.vocab_size;
    int *d_ids; float *d_logits;
    cudaMalloc(&d_ids, B*T*4); cudaMalloc(&d_logits, B*T*V*4);
    cudaMemset(d_ids, 0, B*T*4);
    cudaStream_t s; cudaStreamCreate(&s);
    model_forward_fp32(cfg, w, d_ids, d_logits, B, T, s);
    cudaStreamSynchronize(s);
    fprintf(stderr, "Forward: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(d_ids); cudaFree(d_logits); cudaStreamDestroy(s);
    w.free_all();
    return 0;
}
