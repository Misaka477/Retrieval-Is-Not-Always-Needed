#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "model.h"
#include "infer/infer_base.h"
#include "model.h"

static std::vector<int> read_tokens(const char* path) {
    std::vector<int> tokens;
    FILE* f = fopen(path, "rb");
    if (!f) return tokens;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    if (sz >= 4) { tokens.resize(sz / 4); fread(tokens.data(), 4, tokens.size(), f); }
    fclose(f);
    return tokens;
}

static double lse(const float* x, int n) {
    double mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0.0; for (int i = 0; i < n; i++) s += exp((double)x[i] - mx);
    return mx + log(s);
}

int main(int argc, char** argv) {
    const char *model_path = nullptr, *tokens_path = nullptr;
    int ctx = 1024, stride = 512;
    bool gguf = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1<argc) tokens_path = argv[++i];
        else if (!strcmp(argv[i], "--context") && i+1<argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stride") && i+1<argc) stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gguf")) gguf = true;
        else { fprintf(stderr,"Usage...\n"); return 1; }
    }

    ModelConfig cfg; TensorMap weights;
    if (gguf) { if (!load_gguf_model(model_path, cfg, weights, 0)) return 1; }
    else { if (!load_model(model_path, cfg, weights)) return 1; }
    fprintf(stderr, "%s: %dL dim=%d V=%d\n", gguf?"GGUF":".rinn", cfg.n_layers, cfg.dim, cfg.vocab_size);

    register_all_inferences();
    std::string arch = "gqa";
    if (!cfg.layers.empty() && cfg.layers[0].type.find("deepseek") != std::string::npos) arch = "mla";
    Inference* infer = create_inference(arch);
    if (!infer || !infer->init(cfg, weights)) { fprintf(stderr, "FAIL: init\n"); return 1; }

    auto tokens = read_tokens(tokens_path);
    int N = (int)tokens.size();
    if (N < 3) { fprintf(stderr, "<3 tokens\n"); return 1; }

    if (ctx > cfg.max_seq_len && cfg.max_seq_len > 0) ctx = cfg.max_seq_len;
    if (stride > ctx) stride = ctx;

    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, ctx * sizeof(int));
    float* d_log; cudaMalloc(&d_log, ctx * cfg.vocab_size * sizeof(float));

    std::vector<int> evald(N, 0);
    double total_ll = 0.0; int64_t count = 0;

    for (int start = 0; start < N; start += stride) {
        int end = std::min(start + ctx, N);
        int T = end - start;
        if (T < 2) break;

        cudaMemcpy(d_ids, &tokens[start], T * sizeof(int), cudaMemcpyHostToDevice);
        fprintf(stderr,"  [dbg] forward: T=%d start=%d\n", T, start);
        infer->forward(d_ids, d_log, 1, T, 0, s);
        cudaStreamSynchronize(s);

        std::vector<float> logbuf(T * cfg.vocab_size);
        cudaMemcpy(logbuf.data(), d_log, T * cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int pos = 0; pos < T - 1; pos++) {
            int target = start + pos + 1;
            if (target >= N || evald[target]) continue;
            evald[target] = 1;
            const float* lp = &logbuf[pos * cfg.vocab_size];
            total_ll += lp[tokens[target]] - lse(lp, cfg.vocab_size);
            count++;
        }
    }

    double ppl = exp(-total_ll / count);
    fprintf(stderr, "  [result] total_ll=%.4f count=%ld\n", total_ll, count);
    printf("{\"total_log_prob\":%.10f,\"num_tokens\":%ld,\"ppl\":%.6f}\n", total_ll, count, ppl);

    cudaFree(d_ids); cudaFree(d_log); cudaStreamDestroy(s);
    delete infer;
    return 0;
}
