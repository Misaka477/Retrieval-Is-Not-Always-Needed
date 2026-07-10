#ifndef GGUF_LLAMA_RUNTIME_H
#define GGUF_LLAMA_RUNTIME_H

#include <cstdint>

struct LlamaRuntime;

LlamaRuntime * llama_runtime_load(const char * path, int n_ctx);
void llama_runtime_free(LlamaRuntime * runtime);
int llama_runtime_vocab_size(const LlamaRuntime * runtime);
float * llama_runtime_forward(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens);
void llama_runtime_reset(LlamaRuntime * runtime);
float * llama_runtime_eval(LlamaRuntime * runtime, const int32_t * tokens, int n_tokens, bool all_logits);
float * llama_runtime_last_logits(LlamaRuntime * runtime);

#endif // GGUF_LLAMA_RUNTIME_H
