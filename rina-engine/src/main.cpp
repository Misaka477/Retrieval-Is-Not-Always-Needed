#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "core/config.h"
#include "core/tensor.h"
#include "core/tokenizer.h"
#include "model.h"
#include "infer/infer_base.h"

#ifdef RINA_WITH_PYTORCH
extern "C" void pt_init(const ModelConfig&, const TensorMap&, bool embed=false, bool ln=false,
    bool attn=false, bool cproj=false, bool mlp=false);
extern "C" void pt_forward(const int*, float*, int, int, cudaStream_t);
extern "C" void pt_free();
#endif

static std::mt19937 rng;
static std::string g_prompt_text;

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
    bool use_bf16 = false;
    bool use_gguf = false;
    std::string kv_quant_mode = "fp32";
    bool pre_rope = false;
    struct stat st;
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
        else if (strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "--text") == 0) {
            g_prompt_text = argv[++i];
        }
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
        else if (strcmp(argv[i], "--bf16") == 0) use_bf16 = true;
        else if (strcmp(argv[i], "--qbits") == 0) { if(i+1<argc) i++; }
        else if (strcmp(argv[i], "--gguf") == 0) use_gguf = true;
        else if (strcmp(argv[i], "--kv-quant") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') kv_quant_mode = argv[++i];
            else kv_quant_mode = "q2k_q1v";
        }
        else if (strcmp(argv[i], "--pre-rope") == 0) pre_rope = true;
    }
    if (seed) rng.seed(seed);

    if (!model_path) {
        fprintf(stderr, "Usage: rina_infer --model model.rinn [--prompt \"Hello\"] [--steps N]\n");
        return 1;
    }

    // Load tokenizer early (needed for --prompt)
    RINNModel rinn;
    {
        std::string tok_path = model_path;
        rinn.load(tok_path.c_str());
    }

    if (prompt.empty()) {
        if (g_prompt_text.empty()) {
            fprintf(stderr, "Error: use --prompt \"text\" or --ids \"id id id\"\n");
            return 1;
        }
        // Try C++ tokenizer first
        if (rinn.tokenizer.vocab_size() > 0) {
            prompt = rinn.tokenizer.encode(g_prompt_text);
            fprintf(stderr, "  prompt=\"%s\" → %zu tokens\n", g_prompt_text.c_str(), prompt.size());
        } else {
            fprintf(stderr, "  warning: no tokenizer found, use --ids instead\n");
        }
        fprintf(stderr, "  tokens:");
        for (int t : prompt) fprintf(stderr, " %d", t);
        fprintf(stderr, "\n");
    }

    ModelConfig cfg;
    TensorMap weights;

    bool is_hf = false;
    cfg.kv_quant_mode = kv_quant_mode;
    cfg.use_pre_rope_k = pre_rope;
    if (kv_quant_mode != "fp32")
        fprintf(stderr, "  kv-quant: %s\n", kv_quant_mode.c_str());
    if (use_gguf) {
        fprintf(stderr, "Loading GGUF model: %s\n", model_path);
        if (!load_gguf_model(model_path, cfg, weights, 0)) {  // 0 = all layers
            fprintf(stderr, "Failed to load GGUF model: %s\n", model_path); return 1;
        }
    } else {
        // Auto-detect: HF directory (has config.json) vs .rinn file/directory
        if (::stat(model_path, &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                std::string cfg_path = std::string(model_path) + "/config.json";
                is_hf = (::stat(cfg_path.c_str(), &st) == 0);
            }
        }

        if (is_hf) {
            fprintf(stderr, "Loading HF model: %s\n", model_path);
            if (!load_hf_model(model_path, cfg, weights, 4)) {
                fprintf(stderr, "Failed to load HF model: %s\n", model_path); return 1;
            }
        } else {
            if (!load_model(model_path, cfg, weights)) {
                fprintf(stderr, "Failed to load model: %s\n", model_path); return 1;
            }
        }
    }

    fprintf(stderr, "Model: %s (%d layers, dim=%d, vocab=%d) %s\n",
        cfg.name.c_str(), cfg.n_layers, cfg.dim, cfg.vocab_size,
        use_pytorch ? "[PyTorch ref]" : "[Custom engine]");
    
    // For GGUF: try tokenizer from model dir (same path without .gguf)
    if (use_gguf && rinn.tokenizer.vocab_size() == 0) {
        std::string tok_path = model_path;
        auto dot = tok_path.rfind('.');
        if (dot != std::string::npos) tok_path = tok_path.substr(0, dot);
        rinn.load(tok_path.c_str());
    }
    if(rinn.tokenizer.vocab_size() > 0 && !g_prompt_text.empty()){
        prompt = rinn.tokenizer.encode(g_prompt_text);
        fprintf(stderr, "  prompt=\"%s\" → %zu tokens\n", g_prompt_text.c_str(), prompt.size());
    }

#ifdef RINA_WITH_PYTORCH
    if (use_pytorch) pt_init(cfg, weights, true/*embed*/, true/*ln*/, false/*attn*/, true/*cproj*/, true/*mlp*/);
#endif

    int B = 1, max_seq_len = cfg.max_seq_len > 0 ? std::min(cfg.max_seq_len, 512) : 512;
    int* d_ids; float* d_logits;
    int logit_cap = (int)prompt.size() + steps;
    cudaMalloc(&d_ids, B * (max_seq_len + steps) * sizeof(int));
    cudaMalloc(&d_logits, B * logit_cap * cfg.vocab_size * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyAsync(d_ids, prompt.data(), prompt.size() * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);

    // Create inference engine
    register_all_inferences();
    std::string arch_name = "gqa";
    if (!cfg.layers.empty()) {
        auto& t0 = cfg.layers[0].type;
        if (t0.find("deepseek") != std::string::npos) arch_name = "mla";
    }
    Inference* infer = create_inference(arch_name);
    if (!infer || !infer->init(cfg, weights)) {
        fprintf(stderr, "ERROR: failed to init %s inference\n", arch_name.c_str());
        delete infer;
        return 1;
    }

    // Use KV cache: first call processes all prompt tokens, then process one new token at a time
    std::vector<int> gen = prompt;
    for (int step = 0; step < steps; step++) {
        int T = (step == 0) ? (int)gen.size() : 1;
        int start_pos = (step == 0) ? 0 : (int)gen.size() - 1;
#ifdef RINA_WITH_PYTORCH
        if (use_pytorch) {
            pt_forward(d_ids, d_logits, 1, gen.size(), s);
        } else
#endif
        {
            if (step == 0 && use_bf16) cfg.use_bf16 = true;
            infer->forward(d_ids + start_pos, d_logits, 1, T, start_pos, s);
        }
        cudaStreamSynchronize(s);

        std::vector<float> cpu(cfg.vocab_size);
        cudaMemcpy(cpu.data(), d_logits + (T - 1) * cfg.vocab_size,
                   cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        int next = sample(cpu.data(), cfg.vocab_size, temp, topk, topp, rep, gen, greedy);
        gen.push_back(next);
        cudaMemcpyAsync(d_ids + gen.size() - 1, &next, sizeof(int), cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    }

    delete infer;

    std::vector<int> gen_only(gen.begin() + prompt.size(), gen.end());
    if(rinn.tokenizer.vocab_size() > 0) {
        printf("%s\n", rinn.tokenizer.decode(gen_only).c_str());
    } else {
        printf("tokens:");
        for(auto id: gen_only) printf(" %d", id);
        printf("\n");
    }

    weights.free_all();
#ifdef RINA_WITH_PYTORCH
    if (use_pytorch) pt_free();
#endif
    cudaFree(d_ids); cudaFree(d_logits);
    cudaStreamDestroy(s);
    return 0;
}
