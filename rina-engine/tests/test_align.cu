#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    int B = 1, T = 64;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0) seed = (unsigned int)atoi(argv[++i]);
    }
    if (!model_path) {
        fprintf(stderr, "Usage: test_align --model model.rinn [--batch B] [--seq T]\n");
        return 1;
    }

    ModelConfig cfg;
    TensorMap weights;
    if (!load_model(model_path, cfg, weights)) {
        fprintf(stderr, "Failed to load: %s\n", model_path);
        return 1;
    }

    int V = cfg.vocab_size, d = cfg.dim, n = B * T;
    fprintf(stderr, "Model: %s (%d layers, dim=%d, vocab=%d)\n",
        cfg.name.c_str(), cfg.n_layers, d, V);
    fprintf(stderr, "Align: B=%d T=%d\n", B, T);

    std::mt19937 rng(seed);
    std::vector<int> h_ids(B * (T + 1));

    int *d_ids, *d_targets;
    float *d_logits_v1, *d_logits_v2, *d_loss;
    cudaMalloc(&d_ids, B * (T + 1) * sizeof(int));
    cudaMalloc(&d_targets, B * T * sizeof(int));
    cudaMalloc(&d_logits_v1, n * V * sizeof(float));
    cudaMalloc(&d_logits_v2, n * V * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Generate random inputs
    for (int i = 0; i < B * T; i++) h_ids[i] = rng() % V;
    cudaMemcpyAsync(d_ids, h_ids.data(), B * T * sizeof(int), cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < B * T; i++) h_ids[i] = rng() % V;
    cudaMemcpyAsync(d_targets, h_ids.data(), B * T * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ─── Forward alignment ───
    fprintf(stderr, "\n─── Forward Alignment ───\n");

    // Run v2 first (new implementation) to see its errors independently
    fprintf(stderr, "  Running v2 forward...\n");
    cudaGetLastError();
    model_forward_v2(cfg, weights, d_ids, d_logits_v2, B, T, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err2 = cudaGetLastError();
    fprintf(stderr, "  v2 forward: %s\n", cudaGetErrorString(err2));

    fprintf(stderr, "  Running v1 forward...\n");
    cudaGetLastError();
    model_forward_fp32(cfg, weights, d_ids, d_logits_v1, B, T, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
        fprintf(stderr, "  v1 forward error: %s\n", cudaGetErrorString(err1));

    std::vector<float> cpu1(n * V), cpu2(n * V);
    cudaMemcpy(cpu1.data(), d_logits_v1, n * V * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu2.data(), d_logits_v2, n * V * sizeof(float), cudaMemcpyDeviceToHost);

    double max_diff = 0, sum_diff = 0;
    int max_idx = 0, n_nan = 0, n_inf = 0;
    for (int i = 0; i < n * V; i++) {
        double d = fabs((double)cpu1[i] - (double)cpu2[i]);
        if (d > max_diff) { max_diff = d; max_idx = i; }
        sum_diff += d;
        if (std::isnan(cpu1[i]) || std::isnan(cpu2[i])) n_nan++;
        if (!std::isfinite(cpu1[i]) || !std::isfinite(cpu2[i])) n_inf++;
    }
    fprintf(stderr, "  v1 max=%.6f min=%.6f\n", cpu1[0], cpu1[n*V-1]);
    fprintf(stderr, "  v2 max=%.6f min=%.6f\n", cpu2[0], cpu2[n*V-1]);
    fprintf(stderr, "  max_diff=%.2e at idx=%d\n", max_diff, max_idx);
    fprintf(stderr, "  avg_diff=%.2e\n", sum_diff / (n * V));
    fprintf(stderr, "  nan_count=%d inf_count=%d\n", n_nan, n_inf);

    // ─── Training alignment ───
    fprintf(stderr, "\n─── Training Alignment ───\n");

    float loss_v1 = model_train(cfg, weights, d_ids, d_targets, d_loss, B, T, 0, stream);
    cudaStreamSynchronize(stream);
    fprintf(stderr, "  v1 loss: %.6f\n", loss_v1);

    float loss_v2 = model_train_v2(cfg, weights, d_ids, d_targets, d_loss, B, T, 0, stream);
    cudaStreamSynchronize(stream);
    fprintf(stderr, "  v2 loss: %.6f\n", loss_v2);

    fprintf(stderr, "  loss_diff: %.2e\n", fabs(loss_v1 - loss_v2));

    // ─── Results ───
    bool forward_ok = (max_diff < 1e-5 && n_nan == 0 && n_inf == 0);
    bool train_ok = (fabs(loss_v1 - loss_v2) < 1e-4 && std::isfinite(loss_v1) && std::isfinite(loss_v2));
    fprintf(stderr, "\n─── Results ───\n");
    fprintf(stderr, "  forward_alignment: %s (max_diff=%.2e)\n",
        forward_ok ? "PASS" : "FAIL", max_diff);
    fprintf(stderr, "  train_alignment:   %s (loss_diff=%.2e)\n",
        train_ok ? "PASS" : "FAIL", fabs(loss_v1 - loss_v2));

    cudaFree(d_ids); cudaFree(d_targets);
    cudaFree(d_logits_v1); cudaFree(d_logits_v2); cudaFree(d_loss);
    cudaStreamDestroy(stream);
    weights.free_all();

    return (forward_ok && train_ok) ? 0 : 1;
}
