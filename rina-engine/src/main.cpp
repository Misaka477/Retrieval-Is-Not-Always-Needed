#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#include "core/config.h"
#include "core/tensor.h"
#include "core/tokenizer.h"
#include "model.h"
#include "infer/infer_base.h"
#include "gguf_llama_runtime.h"

#ifdef RINA_WITH_PYTORCH
extern "C" void pt_init(const ModelConfig&, const TensorMap&, bool embed=false, bool ln=false,
    bool attn=false, bool cproj=false, bool mlp=false);
extern "C" void pt_forward(const int*, float*, int, int, cudaStream_t);
extern "C" void pt_free();
#endif

static std::mt19937 rng;
static std::string g_prompt_text;

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static std::string take_utf8_text(std::string& pending, bool final) {
    std::string out;
    size_t i = 0;
    while (i < pending.size()) {
        const unsigned char lead = (unsigned char)pending[i];
        size_t width = 1;
        if (lead < 0x80) width = 1;
        else if (lead >= 0xc2 && lead <= 0xdf) width = 2;
        else if (lead >= 0xe0 && lead <= 0xef) width = 3;
        else if (lead >= 0xf0 && lead <= 0xf4) width = 4;
        else {
            out += "\xef\xbf\xbd";
            i++;
            continue;
        }
        if (i + width > pending.size()) {
            if (final) {
                out += "\xef\xbf\xbd";
                i = pending.size();
            }
            break;
        }
        bool valid = true;
        for (size_t j = 1; j < width; j++) {
            if (((unsigned char)pending[i + j] & 0xc0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (valid && width == 3) {
            const unsigned char second = (unsigned char)pending[i + 1];
            valid = !((lead == 0xe0 && second < 0xa0) || (lead == 0xed && second >= 0xa0));
        } else if (valid && width == 4) {
            const unsigned char second = (unsigned char)pending[i + 1];
            valid = !((lead == 0xf0 && second < 0x90) || (lead == 0xf4 && second >= 0x90));
        }
        if (!valid) {
            out += "\xef\xbf\xbd";
            i++;
            continue;
        }
        out.append(pending, i, width);
        i += width;
    }
    pending.erase(0, i);
    return out;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void normalize_probs(std::vector<std::pair<float,int>>& idx, int nk) {
    float sum = 0.0f;
    for (int i = 0; i < nk; i++) sum += idx[i].first;
    if (sum <= 0.0f) return;
    for (int i = 0; i < nk; i++) idx[i].first /= sum;
}

static bool parse_logit_bias(const char * spec, std::pair<int, float>& out) {
    char * end = nullptr;
    long token = strtol(spec, &end, 10);
    if (!end || *end != ':') return false;
    char * bias_end = nullptr;
    float bias = strtof(end + 1, &bias_end);
    if (!bias_end || *bias_end != '\0') return false;
    out = {(int)token, bias};
    return true;
}

static float biased_logit(float value, int token, const std::vector<std::pair<int, float>>& logit_biases) {
    for (const auto& bias : logit_biases) {
        if (bias.first == token) return value + bias.second;
    }
    return value;
}

static int sample(const float* logits, int n, float temp, int topk, float topp, float minp, float typical,
                  float rep, float freq_penalty, float presence_penalty, int penalty_last_n,
                  const std::vector<std::pair<int, float>>& logit_biases,
                  const std::vector<int>& gen, bool greedy) {
    if (greedy || temp <= 0) {
        float mx = -1e10f; int argmax = 0;
        for (int i = 0; i < n; i++) {
            float v = logits[i];
            if (v > mx) { mx = v; argmax = i; }
        }
        return argmax;
    }

    std::unordered_map<int, int> counts;
    const int history_n = penalty_last_n < 0 ? (int)gen.size() : std::min((int)gen.size(), penalty_last_n);
    const int history_begin = (int)gen.size() - history_n;
    for (int k = history_begin; k < (int)gen.size(); k++) {
        if (gen[k] >= 0 && gen[k] < n) counts[gen[k]]++;
    }

    // 1. Copy + apply penalties + temperature
    std::vector<std::pair<float,int>> idx(n);
    for (int i = 0; i < n; i++) {
        float v = biased_logit(logits[i], i, logit_biases);
        auto count_it = counts.find(i);
        const int count = count_it == counts.end() ? 0 : count_it->second;
        if (count > 0 && rep != 1.0f) v = (v > 0) ? v / rep : v * rep;
        if (count > 0) v -= (float)count * freq_penalty + presence_penalty;
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

    // 4. Min-p: keep tokens whose probability is at least minp * max_probability.
    if (minp > 0.0f && minp < 1.0f && nk > 0) {
        float maxp = 0.0f;
        for (int i = 0; i < nk; i++) maxp = std::max(maxp, idx[i].first);
        const float threshold = maxp * minp;
        bool kept_any = false;
        for (int i = 0; i < nk; i++) {
            if (idx[i].first >= threshold) kept_any = true;
            else idx[i].first = 0.0f;
        }
        if (kept_any) normalize_probs(idx, nk);
    }

    // 5. Top-p (nucleus): zero out tokens below cumulative threshold
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
        if (sum2 > 0.0f) for (int i = 0; i < nk; i++) idx[i].first /= sum2;
    }

    // 6. Locally typical sampling.
    if (typical > 0.0f && typical < 1.0f && nk > 0) {
        double entropy = 0.0;
        for (int i = 0; i < nk; i++) {
            if (idx[i].first > 0.0f) entropy -= idx[i].first * logf(idx[i].first);
        }
        std::vector<int> order(nk);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const float pa = std::max(idx[a].first, 1e-20f);
            const float pb = std::max(idx[b].first, 1e-20f);
            return fabsf(-logf(pa) - (float)entropy) < fabsf(-logf(pb) - (float)entropy);
        });
        std::vector<char> keep(nk, 0);
        float cum = 0.0f;
        for (int rank = 0; rank < nk; rank++) {
            const int j = order[rank];
            if (rank == 0 || cum < typical) {
                keep[j] = 1;
                cum += idx[j].first;
            }
        }
        for (int i = 0; i < nk; i++) if (!keep[i]) idx[i].first = 0.0f;
        normalize_probs(idx, nk);
    }

    // 7. Multinomial over the filtered candidate set.
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cum = 0.0;
    for (int i = 0; i < nk; i++) {
        cum += idx[i].first;
        if (r <= cum) return idx[i].second;
    }
    return idx[std::max(0, nk - 1)].second;
}

static int sample_greedy(const float* logits, int n, const std::vector<std::pair<int, float>>& logit_biases) {
    float mx = -1e10f;
    int argmax = 0;
    for (int i = 0; i < n; i++) {
        float v = biased_logit(logits[i], i, logit_biases);
        if (v > mx) { mx = v; argmax = i; }
    }
    return argmax;
}

static size_t find_stop_pos(const std::string& text, const std::vector<std::string>& stop_strings) {
    size_t best = std::string::npos;
    for (const std::string& stop : stop_strings) {
        if (stop.empty()) continue;
        const size_t pos = text.find(stop);
        if (pos != std::string::npos && (best == std::string::npos || pos < best)) best = pos;
    }
    return best;
}

int main(int argc, char** argv) {
    unsigned int seed = 0;
    const char* model_path = nullptr;
    std::vector<int> prompt;
    int steps = 1;
    int bench_prompt = 0;
    int bench_reps = 3;
    int requested_ctx = 0;
    float temp = 0.8f, topp = 0.9f, minp = 0.0f, typical = 1.0f, rep = 1.1f;
    float freq_penalty = 0.0f, presence_penalty = 0.0f;
    int penalty_last_n = -1;
    int topk = 40;
    bool greedy = true;
    bool use_pytorch = false;
    bool use_bf16 = false;
    bool use_gguf = false;
    bool print_token_ids = false;
    bool jsonl = false;
    bool stream = true;
    bool quiet = false;
    std::string kv_quant_mode = "fp32";
    std::string sampler_backend = "rina";
    std::vector<std::string> stop_strings;
    std::vector<std::pair<int, float>> logit_biases;
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
        else if (strcmp(argv[i], "--steps") == 0) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 || strcmp(argv[i], "--context") == 0) requested_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bench-prompt") == 0) bench_prompt = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bench-reps") == 0) bench_reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0) { temp = atof(argv[++i]); greedy = false; }
        else if (strcmp(argv[i], "--topk") == 0 || strcmp(argv[i], "--top-k") == 0) topk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topp") == 0 || strcmp(argv[i], "--top-p") == 0) topp = atof(argv[++i]);
        else if (strcmp(argv[i], "--minp") == 0 || strcmp(argv[i], "--min-p") == 0) { minp = atof(argv[++i]); greedy = false; }
        else if (strcmp(argv[i], "--typical") == 0 || strcmp(argv[i], "--typical-p") == 0) { typical = atof(argv[++i]); greedy = false; }
        else if (strcmp(argv[i], "--rep") == 0 || strcmp(argv[i], "--repeat-penalty") == 0) rep = atof(argv[++i]);
        else if (strcmp(argv[i], "--frequency-penalty") == 0) freq_penalty = atof(argv[++i]);
        else if (strcmp(argv[i], "--presence-penalty") == 0) presence_penalty = atof(argv[++i]);
        else if (strcmp(argv[i], "--repeat-last-n") == 0 || strcmp(argv[i], "--penalty-last-n") == 0) penalty_last_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0) seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--bf16") == 0) use_bf16 = true;
        else if (strcmp(argv[i], "--qbits") == 0) { if(i+1<argc) i++; }
        else if (strcmp(argv[i], "--gguf") == 0) use_gguf = true;
        else if (strcmp(argv[i], "--print-token-ids") == 0) print_token_ids = true;
        else if (strcmp(argv[i], "--jsonl") == 0) jsonl = true;
        else if (strcmp(argv[i], "--stream") == 0) stream = true;
        else if (strcmp(argv[i], "--no-stream") == 0) stream = false;
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else if (strcmp(argv[i], "--stop") == 0) stop_strings.emplace_back(argv[++i]);
        else if (strcmp(argv[i], "--sampler-backend") == 0) sampler_backend = argv[++i];
        else if (strcmp(argv[i], "--logit-bias") == 0) {
            std::pair<int, float> bias;
            if (!parse_logit_bias(argv[++i], bias)) {
                fprintf(stderr, "Invalid --logit-bias value, expected token_id:bias\n");
                return 1;
            }
            logit_biases.push_back(bias);
        }
        else if (strcmp(argv[i], "--kv-quant") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') kv_quant_mode = argv[++i];
            else kv_quant_mode = "q2k_q1v";
        }
        else if (strcmp(argv[i], "--pre-rope") == 0) pre_rope = true;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: rina_infer --model model.rinn|model.gguf [--gguf] [--prompt text|--ids ids] [--steps N] [--ctx N] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--typical-p P] [--repeat-penalty P] [--frequency-penalty P] [--presence-penalty P] [--repeat-last-n N] [--logit-bias token:bias] [--stop text] [--sampler-backend rina|llama] [--stream|--no-stream] [--print-token-ids] [--jsonl] [--quiet]\n");
            return 1;
        }
    }
    if (sampler_backend != "rina" && sampler_backend != "llama") {
        fprintf(stderr, "Invalid --sampler-backend: %s\n", sampler_backend.c_str());
        return 1;
    }
    rng.seed(seed);
    if (quiet) setenv("RINA_LLAMA_QUIET", "1", 1);

    if (!model_path) {
        fprintf(stderr, "Usage: rina_infer --model model.rinn [--prompt \"Hello\"] [--steps N]\n");
        return 1;
    }

    // Load tokenizer early (needed for --prompt)
    RINNModel rinn;
    if (!use_gguf) {
        std::string tok_path = model_path;
        rinn.load(tok_path.c_str());
    }

    if (prompt.empty()) {
        if (g_prompt_text.empty()) {
            fprintf(stderr, "Error: use --prompt \"text\" or --ids \"id id id\"\n");
            return 1;
        }
        if (use_gguf) {
            if (!quiet) fprintf(stderr, "  gguf prompt text will be tokenized by llama.cpp runtime\n");
        } else {
        // Try C++ tokenizer first
            if (rinn.tokenizer.vocab_size() > 0) {
                prompt = rinn.tokenizer.encode(g_prompt_text);
                fprintf(stderr, "  prompt=\"%s\" -> %zu tokens\n", g_prompt_text.c_str(), prompt.size());
            } else {
                fprintf(stderr, "  warning: no tokenizer found, use --ids instead\n");
            }
            fprintf(stderr, "  tokens:");
            for (int t : prompt) fprintf(stderr, " %d", t);
            fprintf(stderr, "\n");
        }
    }

    ModelConfig cfg;
    TensorMap weights;

    bool is_hf = false;
    cfg.kv_quant_mode = kv_quant_mode;
    cfg.use_pre_rope_k = pre_rope;
    if (kv_quant_mode != "fp32")
        fprintf(stderr, "  kv-quant: %s\n", kv_quant_mode.c_str());
    if (use_gguf) {
        if (!quiet) fprintf(stderr, "Loading GGUF model via llama.cpp CUDA runtime: %s\n", model_path);
        const double t_load0 = now_ms();
        int runtime_ctx = requested_ctx;
        if (runtime_ctx <= 0) {
            int needed = (int)prompt.size() + std::max(0, steps) + 16;
            if (bench_prompt > 0) needed = std::max(needed, bench_prompt);
            if (prompt.empty() && !g_prompt_text.empty()) needed = std::max(needed, (int)g_prompt_text.size() + std::max(0, steps) + 16);
            runtime_ctx = std::max(512, needed);
        }
        LlamaRuntime * rt = llama_runtime_load(model_path, runtime_ctx);
        if (!rt) return 1;
        const double t_load_ms = now_ms() - t_load0;
        if (prompt.empty() && !g_prompt_text.empty()) {
            int n_llama_tokens = 0;
            int32_t * llama_tokens = llama_runtime_tokenize_text(rt, g_prompt_text.c_str(), &n_llama_tokens, true, true);
            if (!llama_tokens || n_llama_tokens <= 0) {
                fprintf(stderr, "llama runtime: failed to tokenize prompt text\n");
                free(llama_tokens);
                llama_runtime_free(rt);
                return 1;
            }
            prompt.assign(llama_tokens, llama_tokens + n_llama_tokens);
            free(llama_tokens);
            if (!quiet) {
                fprintf(stderr, "  gguf prompt=\"%s\" -> %zu tokens\n", g_prompt_text.c_str(), prompt.size());
                fprintf(stderr, "  tokens:");
                for (int t : prompt) fprintf(stderr, " %d", t);
                fprintf(stderr, "\n");
            }
        }
        if (bench_prompt > 0) {
            std::vector<int32_t> bench_ids(bench_prompt, 1);
            if (!prompt.empty()) {
                for (int i = 0; i < bench_prompt; i++) bench_ids[i] = prompt[(size_t)i % prompt.size()];
            }
            float * warmup = llama_runtime_eval(rt, bench_ids.data(), bench_prompt, false);
            free(warmup);
            double total_ms = 0.0;
            for (int rep_i = 0; rep_i < std::max(1, bench_reps); rep_i++) {
                llama_runtime_reset(rt);
                const double t0 = now_ms();
                float * out = llama_runtime_eval(rt, bench_ids.data(), bench_prompt, false);
                total_ms += now_ms() - t0;
                free(out);
            }
            LlamaRuntimePerf perf = llama_runtime_perf(rt);
            const int reps = std::max(1, bench_reps);
            const double avg_ms = total_ms / reps;
            const double tps = avg_ms > 0.0 ? (double)bench_prompt * 1000.0 / avg_ms : 0.0;
            fprintf(stderr, "RINA_PROMPT_BENCH {\"load_ms\":%.3f,\"prompt_tokens\":%d,\"repetitions\":%d,\"avg_ms\":%.3f,\"tokens_per_second\":%.3f,\"llama_prompt_eval_ms\":%.3f,\"llama_prompt_tokens\":%d,\"llama_graph_reused\":%d}\n",
                    t_load_ms, bench_prompt, reps, avg_ms, tps, perf.prompt_eval_ms, perf.prompt_tokens, perf.graph_reused);
            llama_runtime_free(rt);
            return 0;
        }
        std::vector<int32_t> ids(prompt.begin(), prompt.end());
        if (ids.empty()) ids = {1};
        const size_t prefill_tokens = ids.size();
        if (!quiet) fprintf(stderr, "  runtime prefill on %zu tokens\n", ids.size());
        llama_runtime_reset(rt);
        const double t_prefill0 = now_ms();
        const float * logits = llama_runtime_eval_last_view(rt, ids.data(), (int)ids.size());
        const double t_prefill_ms = now_ms() - t_prefill0;
        if (logits) {
            int vs = llama_runtime_vocab_size(rt);
            const float * next_logits = logits;
            if (!quiet) {
                fprintf(stderr, "  logits[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
                        next_logits[0], next_logits[1], next_logits[2], next_logits[3], next_logits[4]);
            }
        }
        double t_sample_ms = 0.0;
        double t_decode_ms = 0.0;
        std::vector<int> gen_history(ids.begin(), ids.end());
        std::vector<int32_t> generated_tokens;
        generated_tokens.reserve(std::max(0, steps));
        std::string utf8_pending;
        std::string generated_text;
        bool stopped = false;
        int decoded_steps = 0;
        LlamaRuntimeSampler * llama_sampler = nullptr;
        if (sampler_backend == "llama") {
            std::vector<int32_t> bias_tokens;
            std::vector<float> bias_values;
            bias_tokens.reserve(logit_biases.size());
            bias_values.reserve(logit_biases.size());
            for (const auto& bias : logit_biases) {
                bias_tokens.push_back(bias.first);
                bias_values.push_back(bias.second);
            }
            llama_sampler = llama_runtime_sampler_create(
                llama_runtime_vocab_size(rt), greedy || temp <= 0.0f, seed, temp, topk, topp, minp, typical,
                rep, freq_penalty, presence_penalty, penalty_last_n,
                bias_tokens.data(), bias_values.data(), (int)bias_tokens.size());
            if (!llama_sampler) {
                fprintf(stderr, "llama runtime: failed to create sampler chain\n");
                llama_runtime_free(rt);
                return 1;
            }
            for (int32_t id : ids) llama_runtime_sampler_accept(llama_sampler, id);
        }
        for (int step = 0; step < steps; step++) {
            if (!logits) {
                llama_runtime_sampler_free(llama_sampler);
                llama_runtime_free(rt);
                return 1;
            }
            int vs = llama_runtime_vocab_size(rt);
            const float * next_logits = logits;
            const double t_sample0 = now_ms();
            int next = 0;
            if (llama_sampler) {
                next = llama_runtime_sampler_sample(rt, llama_sampler);
                if (next < 0) {
                    fprintf(stderr, "llama runtime: sampler chain failed\n");
                    llama_runtime_sampler_free(llama_sampler);
                    llama_runtime_free(rt);
                    return 1;
                }
            } else {
                next = (greedy || temp <= 0.0f)
                    ? sample_greedy(next_logits, vs, logit_biases)
                    : sample(next_logits, vs, temp, topk, topp, minp, typical, rep, freq_penalty, presence_penalty, penalty_last_n, logit_biases, gen_history, false);
            }
            t_sample_ms += now_ms() - t_sample0;
            ids.push_back(next);
            generated_tokens.push_back(next);
            if (jsonl || stream || !stop_strings.empty()) {
                char * piece = llama_runtime_detokenize_token(rt, next, false);
                if (piece) utf8_pending += piece;
                free(piece);
                const std::string text = take_utf8_text(utf8_pending, step + 1 == steps);
                const size_t before_append = generated_text.size();
                generated_text += text;
                const size_t stop_pos = find_stop_pos(generated_text, stop_strings);
                stopped = stop_pos != std::string::npos;
                const std::string visible_text = stopped
                    ? generated_text.substr(before_append, stop_pos > before_append ? stop_pos - before_append : 0)
                    : text;
                if (jsonl) {
                    printf("{\"event\":\"token\",\"index\":%d,\"id\":%d,\"text\":\"%s\",\"logprob\":null}\n",
                           step, next, json_escape(visible_text).c_str());
                } else if (!visible_text.empty() && stop_strings.empty()) {
                    fwrite(visible_text.data(), 1, visible_text.size(), stdout);
                }
                fflush(stdout);
            }
            decoded_steps = step + 1;
            if (stopped) break;
            gen_history.push_back(next);
            if (llama_sampler) llama_runtime_sampler_accept(llama_sampler, next);
            int32_t next_i32 = next;
            const double t_decode0 = now_ms();
            logits = llama_runtime_eval_last_view(rt, &next_i32, 1);
            t_decode_ms += now_ms() - t_decode0;
        }
        llama_runtime_sampler_free(llama_sampler);
        if (decoded_steps > 0) {
            if (!jsonl && !stream) {
                char * text = llama_runtime_detokenize_tokens(rt, generated_tokens.data(), (int)generated_tokens.size(), false);
                if (text) {
                    std::string output = text;
                    const size_t stop_pos = find_stop_pos(output, stop_strings);
                    if (stop_pos != std::string::npos) output.resize(stop_pos);
                    printf("%s\n", output.c_str());
                    fflush(stdout);
                    free(text);
                } else {
                    fprintf(stderr, "llama runtime: failed to detokenize generated tokens\n");
                }
            }
            if (!jsonl && stream) {
                if (!stop_strings.empty()) {
                    const size_t stop_pos = find_stop_pos(generated_text, stop_strings);
                    if (stop_pos != std::string::npos) generated_text.resize(stop_pos);
                    fwrite(generated_text.data(), 1, generated_text.size(), stdout);
                }
                printf("\n");
                fflush(stdout);
            }
            if (print_token_ids) {
                printf("tokens:");
                for (int32_t id : generated_tokens) printf(" %d", id);
                printf("\n");
                fflush(stdout);
            }
        }
        LlamaRuntimePerf perf = llama_runtime_perf(rt);
        const double prefill_tps = t_prefill_ms > 0.0 ? (double)prefill_tokens * 1000.0 / t_prefill_ms : 0.0;
        const double decode_tps = t_decode_ms > 0.0 ? (double)decoded_steps * 1000.0 / t_decode_ms : 0.0;
        if (jsonl) {
            printf("{\"event\":\"perf\",\"prefill_tokens_per_second\":%.3f,\"decode_tokens_per_second\":%.3f,\"decode_tokens\":%d}\n",
                   prefill_tps, decode_tps, decoded_steps);
            fflush(stdout);
        }
        llama_runtime_free(rt);
        fprintf(stderr, "RINA_INFER_PERF {\"load_ms\":%.3f,\"prefill_ms\":%.3f,\"prefill_tokens\":%zu,\"prefill_tokens_per_second\":%.3f,\"decode_ms\":%.3f,\"decode_tokens\":%d,\"decode_tokens_per_second\":%.3f,\"sample_ms\":%.3f,\"llama_prompt_eval_ms\":%.3f,\"llama_prompt_tokens\":%d,\"llama_eval_ms\":%.3f,\"llama_eval_tokens\":%d,\"llama_graph_reused\":%d}\n",
                t_load_ms, t_prefill_ms, prefill_tokens, prefill_tps, t_decode_ms, decoded_steps, decode_tps, t_sample_ms,
                perf.prompt_eval_ms, perf.prompt_tokens, perf.eval_ms, perf.eval_tokens, perf.graph_reused);
        if (!quiet) fprintf(stderr, "llama runtime done\n");
        return 0;
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

        int next = sample(cpu.data(), cfg.vocab_size, temp, topk, topp, minp, typical, rep, freq_penalty, presence_penalty, penalty_last_n, logit_biases, gen, greedy);
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
