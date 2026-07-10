#include "gguf_llama_runtime.h"

#include <llama.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_ctx, 512);
    cparams.n_seq_max = 1;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.offload_kqv = true;
    cparams.op_offload = true;
    cparams.no_perf = true;
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
    fprintf(stderr, "llama runtime CUDA: ctx=%d vocab=%d gpu_offload=%s\n",
            runtime->n_ctx, runtime->vocab_size, llama_supports_gpu_offload() ? "yes" : "no");
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

float * llama_runtime_forward(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens) {
    if (!runtime || !runtime->ctx || !tokens || n_tokens <= 0) return nullptr;
    if (n_tokens > runtime->n_ctx) {
        fprintf(stderr, "llama runtime: n_tokens=%d exceeds n_ctx=%d\n", n_tokens, runtime->n_ctx);
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(runtime->ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(runtime->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama runtime: decode failed: %d\n", rc);
        return nullptr;
    }

    const size_t logits_size = (size_t)n_tokens * runtime->vocab_size * sizeof(float);
    float * result = (float *)malloc(logits_size);
    if (!result) return nullptr;
    for (int i = 0; i < n_tokens; i++) {
        const float * row = llama_get_logits_ith(runtime->ctx, i);
        if (!row) {
            free(result);
            return nullptr;
        }
        memcpy(result + (size_t)i * runtime->vocab_size, row, (size_t)runtime->vocab_size * sizeof(float));
    }
    return result;
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
