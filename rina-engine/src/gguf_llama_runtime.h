#ifndef GGUF_LLAMA_RUNTIME_H
#define GGUF_LLAMA_RUNTIME_H

#include <cstdint>

struct LlamaRuntime;

struct LlamaRuntimePerf {
    double prompt_eval_ms = 0.0;
    double eval_ms = 0.0;
    int32_t prompt_tokens = 0;
    int32_t eval_tokens = 0;
    int32_t graph_reused = 0;
};

LlamaRuntime * llama_runtime_load(const char * path, int n_ctx);
void llama_runtime_free(LlamaRuntime * runtime);
int llama_runtime_vocab_size(const LlamaRuntime * runtime);
int llama_runtime_context_size(const LlamaRuntime * runtime);
LlamaRuntimePerf llama_runtime_perf(const LlamaRuntime * runtime);
bool llama_runtime_add_bos(const LlamaRuntime * runtime);
int32_t llama_runtime_bos_token(const LlamaRuntime * runtime);
float * llama_runtime_forward(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens);
float * llama_runtime_forward_select(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, int logits_from, int n_logits);
bool llama_runtime_score_select(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens,
                                int logits_from, int n_logits, const int32_t * targets,
                                double * total_log_prob, double * total_log_prob_sq);
int32_t * llama_runtime_tokenize_text(LlamaRuntime * runtime, const char * text, int * n_tokens, bool add_special, bool parse_special);
char * llama_runtime_detokenize_token(LlamaRuntime * runtime, int32_t token, bool special);
char * llama_runtime_detokenize_tokens(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, bool special);
void llama_runtime_reset(LlamaRuntime * runtime);
float * llama_runtime_eval(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, bool all_logits);
const float * llama_runtime_eval_last_view(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens);
float * llama_runtime_last_logits(LlamaRuntime * runtime);

#endif // GGUF_LLAMA_RUNTIME_H
