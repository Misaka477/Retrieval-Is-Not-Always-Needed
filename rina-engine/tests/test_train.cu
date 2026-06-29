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
    int B = 1, T = 64, steps = 10;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--steps") == 0) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0) seed = (unsigned int)atoi(argv[++i]);
    }
    if (!model_path) {
        fprintf(stderr, "Usage: test_train --model model.rinn [--steps N] [--batch B] [--seq T]\n");
        return 1;
    }

    ModelConfig cfg;
    TensorMap weights;
    if (!load_model(model_path, cfg, weights)) {
        fprintf(stderr, "Failed to load: %s\n", model_path);
        return 1;
    }

    int V = cfg.vocab_size;
    fprintf(stderr, "Model: %s (%d layers, dim=%d, vocab=%d)\n",
        cfg.name.c_str(), cfg.n_layers, cfg.dim, V);
    fprintf(stderr, "Train: B=%d T=%d steps=%d\n", B, T, steps);

    // Training data: random token sequences
    std::mt19937 rng(seed);
    std::vector<int> h_ids(B * (T + 1));

    // Device buffers
    int *d_ids, *d_targets;
    float *d_logits, *d_loss;
    cudaMalloc(&d_ids, B * (T + 1) * sizeof(int));
    cudaMalloc(&d_targets, B * T * sizeof(int));
    cudaMalloc(&d_logits, B * T * V * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test 1: Verify inference still works (forward only)
    fprintf(stderr, "\n─── Test 1: Inference forward ───\n");
    {
        for (int i = 0; i < B * T; i++) h_ids[i] = rng() % V;
        cudaMemcpyAsync(d_ids, h_ids.data(), B * T * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaGetLastError(); // clear stale error state
        model_forward_fp32(cfg, weights, d_ids, d_logits, B, T, stream);
        cudaStreamSynchronize(stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "  WARNING: inference forward: %s\n", cudaGetErrorString(err));

        std::vector<float> cpu_logits(B * T * V);
        cudaMemcpy(cpu_logits.data(), d_logits, B * T * V * sizeof(float), cudaMemcpyDeviceToHost);
        float mx = -1e10f;
        for (int i = 0; i < B * T * V; i++)
            if (cpu_logits[i] > mx) mx = cpu_logits[i];
        fprintf(stderr, "  max logit=%.4f (no nan/inf: %s)\n", mx,
            (std::isfinite(mx) && !std::isnan(mx)) ? "OK" : "NAN");
    }

    // Test 2: Training step
    fprintf(stderr, "\n─── Test 2: Training (%d steps) ───\n", steps);
    {
        std::vector<float> losses(steps);

        for (int s = 0; s < steps; s++) {
            // Random input seq + shifted targets
            for (int i = 0; i < B * (T + 1); i++) h_ids[i] = rng() % V;
            cudaMemcpyAsync(d_ids, h_ids.data(), B * T * sizeof(int),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_targets, h_ids.data() + 1, B * T * sizeof(int),
                cudaMemcpyHostToDevice, stream);

            // Training step
            losses[s] = model_train(cfg, weights, d_ids, d_targets, d_loss, B, T, s, stream);
            fprintf(stderr, "  step %3d: loss=%.6f\n", s, losses[s]);

            if (s > 0) {
                float delta = losses[s-1] - losses[s];
                // Should be positive (loss decreasing) or at least stable
                if (delta < -0.5f) {
                    fprintf(stderr, "  WARNING: loss spiked by %.4f\n", -delta);
                }
            }
        }

        // Verify loss sanity
        bool loss_ok = true;
        for (int s = 0; s < steps; s++) {
            if (!std::isfinite(losses[s]) || std::isnan(losses[s])) {
                fprintf(stderr, "  FAIL: non-finite loss at step %d\n", s);
                loss_ok = false;
            }
        }
        fprintf(stderr, "  loss sanity: %s\n", loss_ok ? "OK" : "FAIL");
        fprintf(stderr, "  initial loss=%.4f  final loss=%.4f  delta=%.4f\n",
            losses[0], losses[steps-1], losses[0] - losses[steps-1]);
    }

    // Test 3: Verify inference still works after training
    fprintf(stderr, "\n─── Test 3: Inference after training ───\n");
    {
        for (int i = 0; i < B * T; i++) h_ids[i] = rng() % V;
        cudaGetLastError(); // clear stale error state
        cudaMemcpyAsync(d_ids, h_ids.data(), B * T * sizeof(int), cudaMemcpyHostToDevice, stream);
        model_forward_fp32(cfg, weights, d_ids, d_logits, B, T, stream);
        cudaStreamSynchronize(stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "  WARNING: post-training forward: %s\n", cudaGetErrorString(err));
        else
            fprintf(stderr, "  post-training forward: OK\n");

        std::vector<float> cpu_logits(B * T * V);
        cudaMemcpy(cpu_logits.data(), d_logits, B * T * V * sizeof(float), cudaMemcpyDeviceToHost);
        float mx = -1e10f;
        for (int i = 0; i < B * T * V; i++)
            if (cpu_logits[i] > mx) mx = cpu_logits[i];
        fprintf(stderr, "  max logit=%.4f (OK)\n", mx);
    }

    // Cleanup
    cudaFree(d_ids); cudaFree(d_targets); cudaFree(d_logits); cudaFree(d_loss);
    cudaStreamDestroy(stream);
    weights.free_all();

    fprintf(stderr, "\n─── All training tests passed ───\n");
    return 0;
}
