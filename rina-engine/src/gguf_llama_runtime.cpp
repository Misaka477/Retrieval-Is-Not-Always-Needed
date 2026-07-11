#include "gguf_llama_runtime.h"

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static void llama_runtime_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

struct LlamaRuntime {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int vocab_size = 0;
    int n_ctx = 0;
    int n_past = 0;
};

static void llama_runtime_init_once() {
    static bool initialized = false;
    if (!initialized) {
        if (!getenv("RINA_LLAMA_VERBOSE")) {
            llama_log_set(llama_runtime_log_callback, nullptr);
        }
        llama_backend_init();
        initialized = true;
    }
}

static int llama_runtime_threads() {
    const char * env = getenv("RINA_LLAMA_THREADS");
    if (env && atoi(env) > 0) return atoi(env);
    return 6;
}

LlamaRuntime * llama_runtime_load(const char * path, int n_ctx) {
    llama_runtime_init_once();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1;
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.main_gpu = 0;
    mparams.use_mmap = true;
    mparams.use_extra_bufts = true;

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "llama runtime: failed to load model: %s\n", path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? (uint32_t)n_ctx : 4096;
    cparams.n_batch = std::max<uint32_t>(cparams.n_ctx, 512);
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_ctx, 512);
    cparams.n_seq_max = 1;
    cparams.n_threads = llama_runtime_threads();
    cparams.n_threads_batch = llama_runtime_threads();
    cparams.offload_kqv = true;
    cparams.op_offload = true;
    cparams.no_perf = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "llama runtime: failed to create context\n");
        llama_model_free(model);
        return nullptr;
    }

    auto * runtime = new LlamaRuntime();
    runtime->model = model;
    runtime->ctx = ctx;
    runtime->vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    runtime->n_ctx = (int)llama_n_ctx(ctx);
    if (!getenv("RINA_LLAMA_QUIET")) {
        fprintf(stderr, "llama runtime CUDA: ctx=%d vocab=%d gpu_offload=%s\n",
                runtime->n_ctx, runtime->vocab_size, llama_supports_gpu_offload() ? "yes" : "no");
    }
    return runtime;
}

void llama_runtime_free(LlamaRuntime * runtime) {
    if (!runtime) return;
    if (runtime->ctx) llama_free(runtime->ctx);
    if (runtime->model) llama_model_free(runtime->model);
    delete runtime;
}

int llama_runtime_vocab_size(const LlamaRuntime * runtime) {
    return runtime ? runtime->vocab_size : 0;
}

int llama_runtime_context_size(const LlamaRuntime * runtime) {
    return runtime ? runtime->n_ctx : 0;
}

LlamaRuntimePerf llama_runtime_perf(const LlamaRuntime * runtime) {
    LlamaRuntimePerf result;
    if (!runtime || !runtime->ctx) return result;
    llama_perf_context_data data = llama_perf_context(runtime->ctx);
    result.prompt_eval_ms = data.t_p_eval_ms;
    result.eval_ms = data.t_eval_ms;
    result.prompt_tokens = data.n_p_eval;
    result.eval_tokens = data.n_eval;
    result.graph_reused = data.n_reused;
    return result;
}

bool llama_runtime_add_bos(const LlamaRuntime * runtime) {
    if (!runtime || !runtime->model) return false;
    return llama_vocab_get_add_bos(llama_model_get_vocab(runtime->model));
}

int32_t llama_runtime_bos_token(const LlamaRuntime * runtime) {
    if (!runtime || !runtime->model) return -1;
    return llama_vocab_bos(llama_model_get_vocab(runtime->model));
}

int32_t * llama_runtime_tokenize_text(LlamaRuntime * runtime, const char * text, int * n_tokens, bool add_special, bool parse_special) {
    if (n_tokens) *n_tokens = 0;
    if (!runtime || !runtime->model || !text || !n_tokens) return nullptr;

    const size_t text_len = strlen(text);
    if (text_len > (size_t)std::numeric_limits<int32_t>::max()) {
        fprintf(stderr, "llama runtime: text too large to tokenize\n");
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(runtime->model);
    int32_t count = llama_tokenize(vocab, text, (int32_t)text_len, nullptr, 0, add_special, parse_special);
    if (count == std::numeric_limits<int32_t>::min()) {
        fprintf(stderr, "llama runtime: tokenization overflow\n");
        return nullptr;
    }
    if (count < 0) count = -count;
    if (count <= 0) return nullptr;

    int32_t * tokens = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    if (!tokens) return nullptr;

    int32_t check = llama_tokenize(vocab, text, (int32_t)text_len, tokens, count, add_special, parse_special);
    if (check < 0 || check > count) {
        fprintf(stderr, "llama runtime: tokenization failed: %d\n", check);
        free(tokens);
        return nullptr;
    }

    *n_tokens = check;
    return tokens;
}

char * llama_runtime_detokenize_token(LlamaRuntime * runtime, int32_t token, bool special) {
    if (!runtime || !runtime->model) return nullptr;

    const llama_vocab * vocab = llama_model_get_vocab(runtime->model);
    int32_t needed = llama_token_to_piece(vocab, token, nullptr, 0, 0, special);
    if (needed == std::numeric_limits<int32_t>::min()) return nullptr;
    if (needed < 0) needed = -needed;

    char * text = (char *)malloc((size_t)needed + 1);
    if (!text) return nullptr;

    int32_t written = llama_token_to_piece(vocab, token, text, needed + 1, 0, special);
    if (written < 0) {
        free(text);
        return nullptr;
    }
    text[written] = '\0';
    return text;
}

char * llama_runtime_detokenize_tokens(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, bool special) {
    if (!runtime || !runtime->model || (!tokens && n_tokens > 0) || n_tokens < 0) return nullptr;

    if (n_tokens == 0) {
        char * empty = (char *)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    const llama_vocab * vocab = llama_model_get_vocab(runtime->model);
    const bool remove_special = !special;
    const bool unparse_special = special;
    int32_t needed = llama_detokenize(vocab, tokens, n_tokens, nullptr, 0, remove_special, unparse_special);
    if (needed == std::numeric_limits<int32_t>::min()) return nullptr;
    if (needed < 0) needed = -needed;

    char * text = (char *)malloc((size_t)needed + 1);
    if (!text) return nullptr;

    int32_t written = llama_detokenize(vocab, tokens, n_tokens, text, needed + 1, remove_special, unparse_special);
    if (written < 0) {
        free(text);
        return nullptr;
    }
    text[written] = '\0';
    return text;
}

static double llama_runtime_lse(const float * row, int n) {
    double mx = -INFINITY;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = row[i];
        if (v > mx) {
            sum = sum * exp(mx - v) + 1.0;
            mx = v;
        } else {
            sum += exp(v - mx);
        }
    }
    return mx + log(sum);
}

static int llama_runtime_score_threads() {
    static int threads = []() {
        const char * env = getenv("RINA_PPL_THREADS");
        if (env && atoi(env) > 0) return atoi(env);
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        return (int)std::min<unsigned int>(hw, 16);
    }();
    return threads;
}

float * llama_runtime_forward_select(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, int logits_from, int n_logits) {
    if (!runtime || !runtime->ctx || !tokens || n_tokens <= 0) return nullptr;
    if (n_tokens > runtime->n_ctx) {
        fprintf(stderr, "llama runtime: n_tokens=%d exceeds n_ctx=%d\n", n_tokens, runtime->n_ctx);
        return nullptr;
    }
    if (logits_from < 0 || n_logits <= 0 || logits_from + n_logits > n_tokens) {
        fprintf(stderr, "llama runtime: invalid logits range from=%d n=%d tokens=%d\n", logits_from, n_logits, n_tokens);
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(runtime->ctx), true);

    const size_t logits_size = (size_t)n_logits * runtime->vocab_size * sizeof(float);
    float * result = (float *)malloc(logits_size);
    if (!result) return nullptr;
    int out_row = 0;

    const int n_batch = std::min(n_tokens, 64);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int start = 0; start < n_tokens; start += n_batch) {
        const int batch_size = std::min(n_batch, n_tokens - start);
        batch.n_tokens = batch_size;
        for (int i = 0; i < batch_size; i++) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            const int pos = start + i;
            batch.logits[i] = pos >= logits_from && pos < logits_from + n_logits;
        }

        int rc = llama_decode(runtime->ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama runtime: decode failed: %d\n", rc);
            llama_batch_free(batch);
            free(result);
            return nullptr;
        }

        for (int i = 0; i < batch_size; i++) {
            const int pos = start + i;
            if (pos < logits_from || pos >= logits_from + n_logits) continue;
            const float * row = llama_get_logits_ith(runtime->ctx, i);
            if (!row) {
                llama_batch_free(batch);
                free(result);
                return nullptr;
            }
            memcpy(result + (size_t)out_row * runtime->vocab_size, row, (size_t)runtime->vocab_size * sizeof(float));
            out_row++;
        }
    }
    llama_batch_free(batch);
    if (out_row != n_logits) {
        fprintf(stderr, "llama runtime: copied %d logits rows, expected %d\n", out_row, n_logits);
        free(result);
        return nullptr;
    }
    return result;
}

bool llama_runtime_score_select(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens,
                                int logits_from, int n_logits, const int32_t * targets,
                                double * total_log_prob, double * total_log_prob_sq) {
    if (!runtime || !runtime->ctx || !tokens || !targets || !total_log_prob || !total_log_prob_sq || n_tokens <= 0) return false;
    if (n_tokens > runtime->n_ctx) {
        fprintf(stderr, "llama runtime: n_tokens=%d exceeds n_ctx=%d\n", n_tokens, runtime->n_ctx);
        return false;
    }
    if (logits_from < 0 || n_logits <= 0 || logits_from + n_logits > n_tokens) {
        fprintf(stderr, "llama runtime: invalid score range from=%d n=%d tokens=%d\n", logits_from, n_logits, n_tokens);
        return false;
    }

    llama_memory_clear(llama_get_memory(runtime->ctx), true);

    int scored = 0;
    const int n_batch = std::min(n_tokens, 64);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int start = 0; start < n_tokens; start += n_batch) {
        const int batch_size = std::min(n_batch, n_tokens - start);
        batch.n_tokens = batch_size;
        for (int i = 0; i < batch_size; i++) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            const int pos = start + i;
            batch.logits[i] = pos >= logits_from && pos < logits_from + n_logits;
        }

        int rc = llama_decode(runtime->ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama runtime: decode failed: %d\n", rc);
            llama_batch_free(batch);
            return false;
        }

        std::vector<const float *> rows;
        std::vector<int> row_targets;
        rows.reserve(batch_size);
        row_targets.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
            const int pos = start + i;
            if (pos < logits_from || pos >= logits_from + n_logits) continue;
            const int target = targets[pos - logits_from];
            if (target < 0 || target >= runtime->vocab_size) {
                fprintf(stderr, "llama runtime: target token %d outside vocab %d\n", target, runtime->vocab_size);
                llama_batch_free(batch);
                return false;
            }
            const float * row = llama_get_logits_ith(runtime->ctx, i);
            if (!row) {
                llama_batch_free(batch);
                return false;
            }
            rows.push_back(row);
            row_targets.push_back(target);
        }

        const int n_rows = (int)rows.size();
        std::vector<double> log_probs(n_rows, 0.0);
        const int n_threads = std::max(1, std::min(llama_runtime_score_threads(), n_rows));
        if (n_threads == 1) {
            for (int r = 0; r < n_rows; r++) {
                log_probs[r] = (double)rows[r][row_targets[r]] - llama_runtime_lse(rows[r], runtime->vocab_size);
            }
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            for (int t = 0; t < n_threads; t++) {
                workers.emplace_back([&, t]() {
                    for (int r = t; r < n_rows; r += n_threads) {
                        log_probs[r] = (double)rows[r][row_targets[r]] - llama_runtime_lse(rows[r], runtime->vocab_size);
                    }
                });
            }
            for (auto & worker : workers) worker.join();
        }
        for (double log_prob : log_probs) {
            *total_log_prob += log_prob;
            *total_log_prob_sq += log_prob * log_prob;
        }
        scored += n_rows;
    }
    llama_batch_free(batch);
    if (scored != n_logits) {
        fprintf(stderr, "llama runtime: scored %d logits rows, expected %d\n", scored, n_logits);
        return false;
    }
    return true;
}

float * llama_runtime_forward(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens) {
    return llama_runtime_forward_select(runtime, tokens, n_tokens, 0, n_tokens);
}

void llama_runtime_reset(LlamaRuntime * runtime) {
    if (!runtime || !runtime->ctx) return;
    llama_memory_clear(llama_get_memory(runtime->ctx), true);
    runtime->n_past = 0;
}

float * llama_runtime_eval(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, bool all_logits) {
    if (!runtime || !runtime->ctx || !tokens || n_tokens <= 0) return nullptr;
    if (runtime->n_past + n_tokens > runtime->n_ctx) {
        fprintf(stderr, "llama runtime: n_past=%d n_tokens=%d exceeds n_ctx=%d\n",
                runtime->n_past, n_tokens, runtime->n_ctx);
        return nullptr;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = runtime->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_logits || i == n_tokens - 1;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(runtime->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama runtime: decode failed: %d\n", rc);
        return nullptr;
    }
    runtime->n_past += n_tokens;

    const int n_rows = all_logits ? n_tokens : 1;
    const size_t logits_size = (size_t)n_rows * runtime->vocab_size * sizeof(float);
    float * result = (float *)malloc(logits_size);
    if (!result) return nullptr;
    for (int i = 0; i < n_rows; i++) {
        const int idx = all_logits ? i : -1;
        const float * row = llama_get_logits_ith(runtime->ctx, idx);
        if (!row) {
            free(result);
            return nullptr;
        }
        memcpy(result + (size_t)i * runtime->vocab_size, row, (size_t)runtime->vocab_size * sizeof(float));
    }
    return result;
}

const float * llama_runtime_eval_last_view(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens) {
    if (!runtime || !runtime->ctx || !tokens || n_tokens <= 0) return nullptr;
    if (runtime->n_past + n_tokens > runtime->n_ctx) {
        fprintf(stderr, "llama runtime: n_past=%d n_tokens=%d exceeds n_ctx=%d\n",
                runtime->n_past, n_tokens, runtime->n_ctx);
        return nullptr;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = runtime->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == n_tokens - 1;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(runtime->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama runtime: decode failed: %d\n", rc);
        return nullptr;
    }
    runtime->n_past += n_tokens;

    return llama_get_logits_ith(runtime->ctx, -1);
}

float * llama_runtime_last_logits(LlamaRuntime * runtime) {
    if (!runtime || !runtime->ctx) return nullptr;
    const float * row = llama_get_logits_ith(runtime->ctx, -1);
    if (!row) return nullptr;
    const size_t logits_size = (size_t)runtime->vocab_size * sizeof(float);
    float * result = (float *)malloc(logits_size);
    if (!result) return nullptr;
    memcpy(result, row, logits_size);
    return result;
}
