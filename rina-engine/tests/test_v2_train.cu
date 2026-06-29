#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/tmp/gqa_test.rinn";
    int B = 1, T = 64, steps = 5;
    unsigned int seed = 42;
    
    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) { fprintf(stderr,"fail load\n"); return 1; }

    int V = cfg.vocab_size;
    std::mt19937 rng(seed);
    std::vector<int> h_ids(B * (T + 1));
    int *d_ids, *d_targets;
    float *d_logits, *d_loss;
    cudaMalloc(&d_ids, B*(T+1)*sizeof(int));
    cudaMalloc(&d_targets, B*T*sizeof(int));
    cudaMalloc(&d_logits, B*T*V*sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaStream_t stream; cudaStreamCreate(&stream);

    // Test v2 forward
    fprintf(stderr,"v2 forward...\n");
    for(int i=0;i<B*T;i++) h_ids[i]=rng()%V;
    cudaMemcpyAsync(d_ids, h_ids.data(), B*T*sizeof(int), cudaMemcpyHostToDevice, stream);
    model_forward_v2(cfg, w, d_ids, d_logits, B, T, stream);
    cudaStreamSynchronize(stream);
    cudaError_t e = cudaGetLastError();
    fprintf(stderr,"  %s\n", cudaGetErrorString(e));
    
    // Test v2 training
    fprintf(stderr,"v2 training (%d steps)...\n", steps);
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < B * (T + 1); i++) h_ids[i] = rng() % V;
        cudaMemcpyAsync(d_ids, h_ids.data(), B*T*sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_targets, h_ids.data()+1, B*T*sizeof(int), cudaMemcpyHostToDevice, stream);
        float loss = model_train_v2(cfg, w, d_ids, d_targets, d_loss, B, T, s, stream);
        fprintf(stderr,"  step %d: loss=%.6f\n", s, loss);
    }
    
    w.free_all();
    cudaFree(d_ids); cudaFree(d_targets); cudaFree(d_logits); cudaFree(d_loss);
    cudaStreamDestroy(stream);
    return 0;
}
