#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "model.h"

#ifdef RINA_WITH_PYTORCH
extern "C" void pt_init(const ModelConfig&, const TensorMap&, bool embed=false, bool ln=false,
    bool attn=false, bool cproj=false, bool mlp=false);
extern "C" void pt_forward(const int*, float*, int, int, cudaStream_t);
extern "C" void pt_free();
#endif

static std::mt19937 rng;

static int sample(const float* logits, int n, float temp, int topk, float topp, float rep, const std::vector<int>& gen, bool greedy) {
    if (greedy || temp <= 0) {
        float mx = -1e10f; int argmax = 0;
        for (int i = 0; i < n; i++) {
            float v = logits[i];
            if (v > mx) { mx = v; argmax = i; }
        }
        return argmax;
    }

    // 1. Copy + apply repetition penalty + temperature
    std::vector<std::pair<float,int>> idx(n);
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        for (size_t k = 0; k < gen.size(); k++)
            if (gen[k] == i) { v = (v > 0) ? v / rep : v * rep; break; }
        idx[i] = {v / temp, i};
    }

    // 2. Top-k: keep only topk candidates
    int nk = (topk > 0 && topk < n) ? topk : n;
    std::partial_sort(idx.begin(), idx.begin() + nk, idx.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });

    // 3. Softmax over kept candidates
    float maxv = -1e10f;
    for (int i = 0; i < nk; i++) if (idx[i].first > maxv) maxv = idx[i].first;
    float sum = 0;
    for (int i = 0; i < nk; i++) { float e = expf(idx[i].first - maxv); idx[i].first = e; sum += e; }
    for (int i = 0; i < nk; i++) idx[i].first /= sum;

    // 4. Top-p (nucleus): zero out tokens below cumulative threshold
    if (topp > 0.0f && topp < 1.0f) {
        std::sort(idx.begin(), idx.begin() + nk,
                  [](auto& a, auto& b) { return a.first > b.first; });
        float cum = 0;
        for (int i = 0; i < nk; i++) {
            if (cum >= topp && idx[i].first < idx[0].first) idx[i].first = 0;
            else cum += idx[i].first;
        }
        float sum2 = 0;
        for (int i = 0; i < nk; i++) sum2 += idx[i].first;
        for (int i = 0; i < nk; i++) idx[i].first /= sum2;
    }

    // 5. Full-size multinomial (matching PyTorch CPU multinomial)
    // Build cumulative distribution over all n elements
    std::vector<double> full_cum(n + 1, 0.0);
    for (int i = 0; i < n; i++) {
        // find idx for token i in the top-k list
        int ti = -1;
        for (int j = 0; j < nk; j++) if (idx[j].second == i) { ti = j; break; }
        full_cum[i+1] = full_cum[i] + (ti >= 0 ? idx[ti].first : 0.0);
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng) * full_cum[n];
    auto it = std::upper_bound(full_cum.begin(), full_cum.end(), r);
    return std::max(0, (int)(it - full_cum.begin()) - 1);
}

int main(int argc, char** argv) {
    unsigned int seed = 0;
    const char* model_path = nullptr;
    std::vector<int> prompt;
    int steps = 1;
    float temp = 0.8f, topp = 0.9f, rep = 1.1f;
    int topk = 40;
    bool greedy = true;
    bool use_pytorch = false;
    struct { bool embed=false,ln=false,attn=false,cproj=false,mlp=false; } custom;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i],"--pytorch")==0) use_pytorch = true;
        else if (strcmp(argv[i], "--custom-embed") == 0) custom.embed = true;
        else if (strcmp(argv[i], "--custom-ln") == 0) custom.ln = true;
        else if (strcmp(argv[i], "--custom-cproj") == 0) custom.cproj = true;
        else if (strcmp(argv[i], "--custom-mlp") == 0) custom.mlp = true;
        else if (strcmp(argv[i], "--custom-attn") == 0) custom.attn = true;
        else if (strcmp(argv[i], "--ids") == 0) {
            const char* p = argv[++i]; while (*p) {
                while (*p == ' ') p++;
                if (*p == 0) break;
                prompt.push_back(atoi(p));
                while (*p && *p != ' ') p++;
            }
        }
        else if (strcmp(argv[i], "--steps") == 0) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0) { temp = atof(argv[++i]); greedy = false; }
        else if (strcmp(argv[i], "--topk") == 0) topk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topp") == 0) topp = atof(argv[++i]);
        else if (strcmp(argv[i], "--rep") == 0) rep = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0) seed = (unsigned int)atoi(argv[++i]);
    }
    if (seed) rng.seed(seed);

    if (!model_path) { fprintf(stderr, "Usage: rina_infer --model model.rinn --ids \"1 2 3\" [--steps N]\n"); return 1; }

    ModelConfig cfg;
    TensorMap weights;
    if (!load_model(model_path, cfg, weights)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path); return 1;
    }

    fprintf(stderr, "Model: %s (%d layers, dim=%d, vocab=%d) %s\n",
        cfg.name.c_str(), cfg.n_layers, cfg.dim, cfg.vocab_size,
        use_pytorch ? "[PyTorch ref]" : "[Custom engine]");

#ifdef RINA_WITH_PYTORCH
    if (use_pytorch) pt_init(cfg, weights, true/*embed*/, true/*ln*/, false/*attn*/, true/*cproj*/, true/*mlp*/);
#endif

    int B = 1, max_seq_len = cfg.max_seq_len > 0 ? cfg.max_seq_len : 512;
    int* d_ids; float* d_logits;
    cudaMalloc(&d_ids, B * (max_seq_len + steps) * sizeof(int));
    cudaMalloc(&d_logits, B * (max_seq_len + steps) * cfg.vocab_size * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyAsync(d_ids, prompt.data(), prompt.size() * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);

    std::vector<int> gen = prompt;
    for (int step = 0; step < steps; step++) {
#ifdef RINA_WITH_PYTORCH
        if (use_pytorch) {
            pt_forward(d_ids, d_logits, 1, gen.size(), s);
        } else
#endif
        {
            model_forward_fp32(cfg, weights, d_ids, d_logits, 1, gen.size(), s);
        }
        cudaStreamSynchronize(s);

        std::vector<float> cpu(cfg.vocab_size);
        cudaMemcpy(cpu.data(), d_logits + (gen.size() - 1) * cfg.vocab_size,
                   cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        int next = sample(cpu.data(), cfg.vocab_size, temp, topk, topp, rep, gen, greedy);
        gen.push_back(next);
        cudaMemcpyAsync(d_ids + gen.size() - 1, &next, sizeof(int), cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    }

    for (size_t i = prompt.size(); i < gen.size(); i++)
        printf("%s%d", i==prompt.size()?"":" ", gen[i]);
    printf("\n");

    weights.free_all();
#ifdef RINA_WITH_PYTORCH
    if (use_pytorch) pt_free();
#endif
    cudaFree(d_ids); cudaFree(d_logits);
    cudaStreamDestroy(s);
    return 0;
}
