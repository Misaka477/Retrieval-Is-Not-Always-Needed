#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include "model.h"
#include "infer/infer_base.h"
#include "core/tokenizer.h"
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

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void accumulate_ppl(const float * logits, int vocab_size, const int * tokens, int n_token,
                           double & total_ll, double & total_ll2, int64_t & count) {
    for (int i = 0; i < n_token; i++) {
        const float * row = logits + (size_t)i * vocab_size;
        const double log_prob = row[tokens[i + 1]] - lse(row, vocab_size);
        total_ll += log_prob;
        total_ll2 += log_prob * log_prob;
        count++;
    }
}

int main(int argc, char** argv) {
    const char *model_path = nullptr, *tokens_path = nullptr, *dump_logits_path = nullptr;
    int ctx = 1024, stride = 0;
    bool gguf = false;
    bool input_text = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1<argc) tokens_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1<argc) { tokens_path = argv[++i]; input_text = true; }
        else if (!strcmp(argv[i], "--context") && i+1<argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stride") && i+1<argc) stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gguf")) gguf = true;
        else if (!strcmp(argv[i], "--dump-logits") && i+1<argc) dump_logits_path = argv[++i];
        else { fprintf(stderr,"Usage: --model model --tokens file.bin|--text file.txt [--context N] [--stride N] [--gguf] [--dump-logits file]\n"); return 1; }
    }

    bool runtime = gguf;
    ModelConfig cfg; TensorMap weights;
    LlamaRuntime * llama_rt = nullptr;
    Inference* infer = nullptr;
    int chunks_done = 0;
    double runtime_eval_ms = 0.0;
    if (runtime) {
        int runtime_ctx = stride > 0 ? ctx + stride : ctx;
        llama_rt = llama_runtime_load(model_path, runtime_ctx);
        if (!llama_rt) return 1;
        if (stride > 0) ctx = llama_runtime_context_size(llama_rt);
        cfg.max_seq_len = ctx;
        cfg.vocab_size = llama_runtime_vocab_size(llama_rt);
        fprintf(stderr, "GGUF llama runtime: V=%d\n", cfg.vocab_size);
    } else {
        if (!load_model(model_path, cfg, weights)) return 1;
        fprintf(stderr, ".rinn: %dL dim=%d V=%d\n", cfg.n_layers, cfg.dim, cfg.vocab_size);

        register_all_inferences();
        std::string arch = "gqa";
        if (!cfg.layers.empty() && cfg.layers[0].type.find("deepseek") != std::string::npos) arch = "mla";
        infer = create_inference(arch);
        if (!infer || !infer->init(cfg, weights)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    }

    RINNModel rinn;
    if (!runtime) {
        rinn.load(model_path);
    }

    std::vector<int> tokens;
    if (tokens_path) {
        std::string text = read_text(tokens_path);
        bool is_text = !text.empty() && text.find('\0') == std::string::npos;
        if (runtime && input_text && is_text) {
            int n_llama_tokens = 0;
            int32_t * llama_tokens = llama_runtime_tokenize_text(llama_rt, text.c_str(), &n_llama_tokens, true, true);
            if (!llama_tokens || n_llama_tokens <= 0) {
                fprintf(stderr, "  error: llama.cpp tokenizer failed\n");
                free(llama_tokens);
                return 1;
            }
            tokens.assign(llama_tokens, llama_tokens + n_llama_tokens);
            free(llama_tokens);
            fprintf(stderr, "  gguf text (%zu bytes) -> %zu tokens\n", text.size(), tokens.size());
        } else if (!input_text && is_text) {
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
    if (!runtime) {
        cudaStreamCreate(&s);
        cudaMalloc(&d_ids, ctx * sizeof(int));
        cudaMalloc(&d_log, ctx * cfg.vocab_size * sizeof(float));
    }

    double total_ll = 0.0, total_ll2 = 0.0; int64_t count = 0;
    FILE * dump_logits = dump_logits_path ? fopen(dump_logits_path, "wb") : nullptr;
    if (dump_logits_path && !dump_logits) { fprintf(stderr, "Cannot open logits dump: %s\n", dump_logits_path); return 1; }

    if (runtime) {
        const bool add_bos = llama_runtime_add_bos(llama_rt);
        const int32_t bos = llama_runtime_bos_token(llama_rt);
        const int score_from = stride > 0 ? ctx - stride - 1 : ctx / 2;
        if (score_from < 0 || score_from >= ctx - 1) {
            fprintf(stderr, "Invalid stride=%d for context=%d\n", stride, ctx);
            return 1;
        }
        if (N < 2 * ctx) {
            fprintf(stderr, "warning: llama.cpp perplexity expects at least %d tokens for context %d; got %d\n", 2 * ctx, ctx, N);
        }

        const int n_chunk = stride > 0 ? std::max(0, (N - ctx + stride - 1) / stride) : N / ctx;
        fprintf(stderr, "  llama-style ppl: chunks=%d ctx=%d stride=%d score_from=%d\n", n_chunk, ctx, stride, score_from);
        const double t_runtime0 = now_ms();
        for (int i = 0; i < n_chunk; i++) {
            const int start = stride > 0 ? i * stride : i * ctx;
            const int end = start + ctx;
            if (end > N) break;

            std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + end);
            if (add_bos && !ids.empty()) ids[0] = bos;
            fprintf(stderr,"  forward: T=%d start=%d\n", ctx, start);

            const int n_score = ctx - score_from - 1;
            const double t_eval0 = now_ms();
            float * out = dump_logits
                ? llama_runtime_forward(llama_rt, ids.data(), ctx)
                : llama_runtime_forward_select(llama_rt, ids.data(), ctx, score_from, n_score);
            runtime_eval_ms += now_ms() - t_eval0;
            if (!out) return 1;
            if (dump_logits) fwrite(out, sizeof(float), (size_t)ctx * cfg.vocab_size, dump_logits);
            accumulate_ppl(dump_logits ? out + (size_t)score_from * cfg.vocab_size : out, cfg.vocab_size,
                           tokens.data() + start + score_from, n_score,
                           total_ll, total_ll2, count);
            free(out);
            chunks_done++;
        }
        const double wall_ms = now_ms() - t_runtime0;
        const double chunk_tps = wall_ms > 0.0 ? (double)chunks_done * 1000.0 / wall_ms : 0.0;
        const double eval_tps = runtime_eval_ms > 0.0 ? (double)chunks_done * ctx * 1000.0 / runtime_eval_ms : 0.0;
        const double scored_tps = wall_ms > 0.0 ? (double)count * 1000.0 / wall_ms : 0.0;
        fprintf(stderr, "RINA_PPL_PERF {\"chunks\":%d,\"ctx\":%d,\"wall_ms\":%.3f,\"eval_ms\":%.3f,\"chunks_per_second\":%.3f,\"eval_tokens_per_second\":%.3f,\"scored_tokens_per_second\":%.3f}\n",
                chunks_done, ctx, wall_ms, runtime_eval_ms, chunk_tps, eval_tps, scored_tps);
    } else {
        std::vector<int> evald(N, 0);
        if (stride <= 0) stride = ctx;
        for (int start = 0; start < N; start += stride) {
            int end = std::min(start + ctx, N);
            int T = end - start;
            if (T < 2) break;

            fprintf(stderr,"  forward: T=%d start=%d\n", T, start);

            std::vector<float> logbuf(T * cfg.vocab_size);
            cudaMemcpy(d_ids, &tokens[start], T * sizeof(int), cudaMemcpyHostToDevice);
            infer->forward(d_ids, d_log, 1, T, 0, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(logbuf.data(), d_log, T * cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

            if (dump_logits) {
                fwrite(logbuf.data(), sizeof(float), logbuf.size(), dump_logits);
            }

            for (int pos = 0; pos < T - 1; pos++) {
                int target = start + pos + 1;
                if (target >= N || evald[target]) continue;
                evald[target] = 1;
                const float* lp = &logbuf[pos * cfg.vocab_size];
                const double log_prob = lp[tokens[target]] - lse(lp, cfg.vocab_size);
                total_ll += log_prob;
                total_ll2 += log_prob * log_prob;
                count++;
            }
        }
    }

    if (count <= 0) { fprintf(stderr, "No tokens scored\n"); return 1; }
    double avg_ll = total_ll / count;
    double variance = total_ll2 / count - avg_ll * avg_ll;
    double ppl = exp(-avg_ll);
    double ppl_std = variance > 0.0 && count > 1 ? sqrt(variance / (count - 1)) * ppl : 0.0;
    fprintf(stderr, "  [result] PPL=%.4f +/- %.5f (total_ll=%.4f count=%ld)\n", ppl, ppl_std, total_ll, count);
    printf("{\"total_log_prob\":%.10f,\"num_tokens\":%ld,\"ppl\":%.6f,\"ppl_std\":%.6f,\"chunks\":%d,\"eval_ms\":%.3f,\"eval_tokens_per_second\":%.3f}\n",
           total_ll, count, ppl, ppl_std, chunks_done, runtime_eval_ms,
           runtime_eval_ms > 0.0 ? (double)chunks_done * ctx * 1000.0 / runtime_eval_ms : 0.0);

    if (dump_logits) fclose(dump_logits);
    if (llama_rt) llama_runtime_free(llama_rt);
    if (!runtime) {
        cudaFree(d_ids); cudaFree(d_log); cudaStreamDestroy(s);
    }
    delete infer;
    return 0;
}
