#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>

#include "model.h"
#include "infer/infer_base.h"
#include "core/tokenizer.h"
#include "gguf_rina_bridge.h"
#include "gguf_llama_runtime.h"

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

static std::string read_text(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    std::string s(sz, '\0');
    fread(&s[0], 1, sz, f);
    fclose(f);
    return s;
}

static std::vector<int> parse_token_ids(const std::string & text) {
    std::vector<int> tokens;
    const char * p = text.c_str();
    char * end = nullptr;
    while (*p) {
        long v = strtol(p, &end, 10);
        if (end == p) {
            p++;
            continue;
        }
        tokens.push_back((int)v);
        p = end;
    }
    return tokens;
}

static double lse(const float* x, int n) {
    double mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0.0; for (int i = 0; i < n; i++) s += exp((double)x[i] - mx);
    return mx + log(s);
}

int main(int argc, char** argv) {
    const char *model_path = nullptr, *tokens_path = nullptr, *dump_logits_path = nullptr;
    int ctx = 1024, stride = 512;
    bool gguf = false;
    bool legacy_gguf = false;
    bool bridge = false;
    bool input_text = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1<argc) tokens_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1<argc) { tokens_path = argv[++i]; input_text = true; }
        else if (!strcmp(argv[i], "--context") && i+1<argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stride") && i+1<argc) stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gguf")) gguf = true;
        else if (!strcmp(argv[i], "--bridge")) bridge = true;
        else if (!strcmp(argv[i], "--legacy-gguf")) legacy_gguf = true;
        else if (!strcmp(argv[i], "--dump-logits") && i+1<argc) dump_logits_path = argv[++i];
        else { fprintf(stderr,"Usage: --model model --tokens file.bin|--text file.txt [--context N] [--stride N] [--gguf] [--bridge] [--legacy-gguf] [--dump-logits file]\n"); return 1; }
    }

    bool runtime = gguf && !legacy_gguf && !bridge;
    ModelConfig cfg; TensorMap weights;
    BridgeModel bridge_model;
    LlamaRuntime * llama_rt = nullptr;
    Inference* infer = nullptr;
    if (runtime) {
        llama_rt = llama_runtime_load(model_path, ctx);
        if (!llama_rt) return 1;
        cfg.max_seq_len = ctx;
        cfg.vocab_size = llama_runtime_vocab_size(llama_rt);
        fprintf(stderr, "GGUF llama runtime: V=%d\n", cfg.vocab_size);
    } else if (bridge) {
        if (!bridge_load_model(model_path, bridge_model)) return 1;
        cfg.max_seq_len = bridge_model.config.max_seq_len;
        cfg.vocab_size = bridge_model.config.vocab_size;
        fprintf(stderr, "GGUF bridge: %dL dim=%d V=%d\n", bridge_model.config.n_layers, bridge_model.config.dim, cfg.vocab_size);
    } else {
        if (gguf) {
            fprintf(stderr, "WARNING: --legacy-gguf uses the old fake ggml_tensor path and is known to produce incorrect PPL. Use --gguf without --legacy-gguf for the bridge path.\n");
            if (!load_gguf_model(model_path, cfg, weights, 0)) return 1;
        } else {
            if (!load_model(model_path, cfg, weights)) return 1;
        }
        fprintf(stderr, "%s: %dL dim=%d V=%d\n", gguf?"GGUF legacy":".rinn", cfg.n_layers, cfg.dim, cfg.vocab_size);

        register_all_inferences();
        std::string arch = "gqa";
        if (!cfg.layers.empty() && cfg.layers[0].type.find("deepseek") != std::string::npos) arch = "mla";
        infer = create_inference(arch);
        if (!infer || !infer->init(cfg, weights)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    }

    // Load tokenizer from model directory (for --text input)
    RINNModel rinn;
    rinn.load(model_path);

    std::vector<int> tokens;
    if (tokens_path) {
        std::string text = read_text(tokens_path);
        bool is_text = !text.empty() && text.find('\0') == std::string::npos;
        if (!input_text && is_text) {
            tokens = parse_token_ids(text);
            fprintf(stderr, "  ascii tokens: %zu\n", tokens.size());
        } else if (is_text && rinn.tokenizer.vocab_size() > 0) {
            tokens = rinn.tokenizer.encode(text);
            fprintf(stderr, "  text (%zu bytes) → %zu tokens\n", text.size(), tokens.size());
        } else if (is_text) {
            fprintf(stderr, "  error: text input requires a tokenizer (model path must have tokenizer.json)\n");
            return 1;
        } else {
            tokens = read_tokens(tokens_path);
            fprintf(stderr, "  binary tokens: %zu\n", tokens.size());
        }
    }
    if (tokens.empty()) { fprintf(stderr, "No tokens\n"); return 1; }
    if (tokens.size() < 3) { fprintf(stderr, "<3 tokens\n"); return 1; }

    int N = (int)tokens.size();
    if (ctx > cfg.max_seq_len && cfg.max_seq_len > 0) ctx = cfg.max_seq_len;
    if (stride > ctx) stride = ctx;

    cudaStream_t s = nullptr;
    int* d_ids = nullptr;
    float* d_log = nullptr;
    if (!bridge && !runtime) {
        cudaStreamCreate(&s);
        cudaMalloc(&d_ids, ctx * sizeof(int));
        cudaMalloc(&d_log, ctx * cfg.vocab_size * sizeof(float));
    }

    std::vector<int> evald(N, 0);
    double total_ll = 0.0; int64_t count = 0;
    FILE * dump_logits = dump_logits_path ? fopen(dump_logits_path, "wb") : nullptr;
    if (dump_logits_path && !dump_logits) { fprintf(stderr, "Cannot open logits dump: %s\n", dump_logits_path); return 1; }

    for (int start = 0; start < N; start += stride) {
        int end = std::min(start + ctx, N);
        int T = end - start;
        if (T < 2) break;

        fprintf(stderr,"  forward: T=%d start=%d\n", T, start);

        std::vector<float> logbuf(T * cfg.vocab_size);
        if (runtime) {
            std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + end);
            float * out = llama_runtime_forward(llama_rt, ids.data(), T);
            if (!out) return 1;
            memcpy(logbuf.data(), out, logbuf.size() * sizeof(float));
            free(out);
        } else if (bridge) {
            std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + end);
            float * out = bridge_forward(bridge_model, ids.data(), T);
            if (!out) return 1;
            memcpy(logbuf.data(), out, logbuf.size() * sizeof(float));
            free(out);
        } else {
            cudaMemcpy(d_ids, &tokens[start], T * sizeof(int), cudaMemcpyHostToDevice);
            infer->forward(d_ids, d_log, 1, T, 0, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(logbuf.data(), d_log, T * cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        }

        if (dump_logits) {
            fwrite(logbuf.data(), sizeof(float), logbuf.size(), dump_logits);
        }

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
    fprintf(stderr, "  [result] PPL=%.4f (total_ll=%.4f count=%ld)\n", ppl, total_ll, count);
    printf("{\"total_log_prob\":%.10f,\"num_tokens\":%ld,\"ppl\":%.6f}\n", total_ll, count, ppl);

    if (dump_logits) fclose(dump_logits);
    if (llama_rt) llama_runtime_free(llama_rt);
    if (!bridge && !runtime) {
        cudaFree(d_ids); cudaFree(d_log); cudaStreamDestroy(s);
    }
    delete infer;
    return 0;
}
