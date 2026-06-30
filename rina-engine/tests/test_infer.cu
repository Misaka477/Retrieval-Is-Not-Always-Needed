#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: test_infer model.rinn \"prompt tokens...\" [steps]\n"); return 1; }
    const char* model_path = argv[1];
    const char* prompt_str = argv[2];
    int max_steps = argc > 3 ? atoi(argv[3]) : 32;

    // Parse prompt
    std::vector<int> prompt;
    const char* p = prompt_str;
    while (*p) {
        while (*p == ' ') p++; if (!*p) break;
        prompt.push_back(atoi(p));
        while (*p && *p != ' ') p++;
    }
    fprintf(stderr, "Prompt: %zu tokens\n", prompt.size());

    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) { fprintf(stderr, "load fail\n"); return 1; }
    auto layers = build_layers(cfg, w);
    if (layers.empty()) { fprintf(stderr, "build layers fail\n"); return 1; }

    int B = 1, d = cfg.dim, V = cfg.vocab_size;
    int T = (int)prompt.size() + max_steps;
    int n = B * T;

    int ws = 0, total = 0;
    for (auto& l : layers) {
        int w = l->workspace_per_token(d, cfg.n_heads, cfg.head_dim); if(w>ws)ws=w;
        total += l->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    }
    BufferManager bufs;
    // Need extra space for generation (T tokens)
    bufs.alloc_fwd(n, d, ws, 8192, V, total);
    if (!bufs.fwd.h) { fprintf(stderr, "alloc fail\n"); return 1; }

    cudaStream_t s; cudaStreamCreate(&s);
    int* d_ids; cudaMalloc(&d_ids, T * sizeof(int));
    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    const float* lm_w = (const float*)w.get("lm_head.weight")->data;

    // Begin autoregressive generation
    std::vector<int> tokens = prompt;
    for (int step = 0; step < max_steps; step++) {
        int cur_T = (int)tokens.size();
        cudaMemcpyAsync(d_ids, tokens.data(), cur_T * sizeof(int), cudaMemcpyHostToDevice, s);

        // Forward
        float* h = bufs.fwd.h;
        launch_embedding_fp32(wte, d_ids, h, B, cur_T, d, s);

        float* base_save = bufs.fwd.save;
        for (int l = 0; l < (int)layers.size(); l++) {
            bufs.fwd.save = base_save + layers[l]->save_offset * cur_T;
            layers[l]->forward(h, bufs.fwd, B, cur_T, s);
        }
        bufs.fwd.save = base_save;

        // Final norm + lm_head
        auto* ln_f = w.get("transformer.ln_f.weight");
        if (ln_f) {
            extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
            launch_rms_norm_fp32(h, (const float*)ln_f->data, cur_T, d, 1e-5f, s);
        }
        launch_linear_fp32(h, lm_w, bufs.fwd.lm, cur_T, V, d, s);
        cudaStreamSynchronize(s);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "step %d: %s\n", step, cudaGetErrorString(e)); break; }

        // Get last token logits and sample
        std::vector<float> last_logits(V);
        cudaMemcpy(last_logits.data(), bufs.fwd.lm + (cur_T - 1) * V, V * sizeof(float), cudaMemcpyDeviceToHost);

        // Argmax sampling
        int next = 0;
        float max_v = -1e10f;
        for (int i = 0; i < V; i++) {
            if (last_logits[i] > max_v) { max_v = last_logits[i]; next = i; }
        }

        tokens.push_back(next);
        fprintf(stderr, "step %2d: token=%-6d", step, next);

        // Print some tokens as chars if printable
        if (next >= 32 && next < 127) fprintf(stderr, " '%c'", (char)next);
        fprintf(stderr, "\n");

        if (next == 2) { fprintf(stderr, "  <EOS>\n"); break; }  // EOS for Llama
    }

    fprintf(stderr, "\nGenerated %zu tokens total\n", tokens.size());
    w.free_all(); bufs.free_all(); cudaFree(d_ids); cudaStreamDestroy(s);
    return 0;
}
