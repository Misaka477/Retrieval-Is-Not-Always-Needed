// test_bf16_align.cu — Compare fp32 vs bf16 forward pass on the same model
// Usage: test_bf16_align model.rinn "id1 id2 ..." [tolerance]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_dispatch(const void*,QuantType,const float*,float*,int,int,int,cudaStream_t);

// Build layers with modified type — replace "standard_attention" with replacement_type
static std::vector<std::unique_ptr<Layer>> build_layers_with_type(
    const ModelConfig& cfg, const TensorMap& weights, const std::string& replacement_type) {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.reserve(cfg.n_layers);
    int accum_offset = 0;
    for (int l = 0; l < cfg.n_layers; l++) {
        std::string type;
        if (l < (int)cfg.layers.size()) {
            type = cfg.layers[l].type;
            // Replace standard_attention with the requested type
            if (type == "standard_gqa" || type == "standard_attention" || type == "gqa")
                type = replacement_type;
        } else {
            type = "inertia_wave_ssm";
        }
        auto layer = std::unique_ptr<Layer>(create_layer_by_type(type));
        if (!layer) {
            fprintf(stderr, "build: unknown type '%s' at %d\n", type.c_str(), l);
            return {};
        }
        if (!layer->init(cfg, weights, l)) {
            fprintf(stderr, "build: init fail for %d '%s'\n", l, type.c_str());
            return {};
        }
        layer->save_offset = accum_offset;
        accum_offset += layer->saved_per_token(cfg.dim, cfg.n_heads, cfg.head_dim);
        layers.push_back(std::move(layer));
    }
    return layers;
}

static float run_forward(ModelConfig& cfg, TensorMap& w,
    const int* d_ids, int B, int T,
    std::vector<std::unique_ptr<Layer>>& layers,
    BufferManager& bufs, cudaStream_t s,
    float* out_logits) {
    int n = B * T, d = cfg.dim, V = cfg.vocab_size;

    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte, d_ids, bufs.fwd.h, B, T, d, s);

    float* base_save = bufs.fwd.save;
    for (int l = 0; l < (int)layers.size(); l++) {
        bufs.fwd.save = base_save + layers[l]->save_offset * n;
        layers[l]->forward(bufs.fwd.h, bufs.fwd, B, T, s);
    }
    bufs.fwd.save = base_save;

    auto* ln_f = w.get("transformer.ln_f.weight");
    if (ln_f) {
        launch_rms_norm_fp32(bufs.fwd.h, (const float*)ln_f->data, n, d, 1e-5f, s);
    }

    auto* lm_t = w.get("lm_head.weight");
    if (lm_t) {
        launch_linear_dispatch(lm_t->data, lm_t->quant_type, bufs.fwd.h, bufs.fwd.lm, n, V, d, s);
    } else {
        extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
        launch_linear_fp32(bufs.fwd.h, wte, bufs.fwd.lm, n, V, d, s);
    }

    cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr,"forward error: %s\n",cudaGetErrorString(e)); return -1.0f; }

    cudaMemcpy(out_logits, bufs.fwd.lm, (size_t)n * V * sizeof(float), cudaMemcpyDeviceToHost);
    return 0.0f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,"Usage: test_bf16_align model.rinn \"id1 id2 ...\" [tolerance]\n");
        return 1;
    }
    const char* model_path = argv[1];
    const char* ids_str = argv[2];
    float tolerance = (argc > 3) ? (float)atof(argv[3]) : 1e-2f;

    std::vector<int> ids;
    const char* p = ids_str;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        ids.push_back(atoi(p));
        while (*p && *p != ' ') p++;
    }
    fprintf(stderr, "Input tokens: %zu\n", ids.size());

    ModelConfig cfg; TensorMap w;
    if (!load_model(model_path, cfg, w)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }
    fprintf(stderr, "Model: %s (%d layers, dim=%d, n_heads=%d, n_kv=%d)\n",
        cfg.name.c_str(), cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads);

    int B = 1, T = (int)ids.size(), n = B * T, d = cfg.dim, V = cfg.vocab_size;

    // Allocate once, build layers twice
    // Build fp32 layers first to compute workspace
    auto fp32_layers = build_layers_with_type(cfg, w, "gqa");
    if (fp32_layers.empty()) { fprintf(stderr, "fp32 build failed\n"); return 1; }
    auto bf16_layers = build_layers_with_type(cfg, w, "gqa_bf16");
    if (bf16_layers.empty()) { fprintf(stderr, "bf16 build failed\n"); return 1; }

    int ws = 0, total = 0;
    for (auto& l : fp32_layers) {
        int w = l->workspace_per_token(d, cfg.n_heads, cfg.head_dim);
        if (w > ws) ws = w;
        total += l->saved_per_token(d, cfg.n_heads, cfg.head_dim);
    }
    fprintf(stderr, "Workspace per token: %d, saved per layer total: %d\n", ws, total);

    // Use two separate buffer managers to avoid interference
    BufferManager bufs_fp32, bufs_bf16;
    bufs_fp32.alloc_fwd(n, d, ws, 8192, V, total);
    bufs_bf16.alloc_fwd(n, d, ws, 8192, V, total);
    if (!bufs_fp32.fwd.h || !bufs_bf16.fwd.h) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    cudaStream_t s_fp32, s_bf16;
    cudaStreamCreate(&s_fp32);
    cudaStreamCreate(&s_bf16);

    int* d_ids;
    cudaMalloc(&d_ids, n * sizeof(int));
    std::vector<int> h_ids(T);
    for (int i = 0; i < T; i++) h_ids[i] = i < (int)ids.size() ? ids[i] : 0;
    cudaMemcpy(d_ids, h_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate output buffers
    std::vector<float> fp32_logits(n * V);
    std::vector<float> bf16_logits(n * V);

    fprintf(stderr, "Running fp32 forward...\n");
    float err = run_forward(cfg, w, d_ids, B, T, fp32_layers, bufs_fp32, s_fp32, fp32_logits.data());
    if (err < 0) { fprintf(stderr, "fp32 forward failed\n"); return 1; }

    // Get last token
    int T_last = T - 1;
    std::vector<float> fp32_last(V);
    std::vector<float> bf16_last(V);
    memcpy(fp32_last.data(), fp32_logits.data() + T_last * V, V * sizeof(float));

    fprintf(stderr, "Running bf16 forward...\n");
    err = run_forward(cfg, w, d_ids, B, T, bf16_layers, bufs_bf16, s_bf16, bf16_logits.data());
    if (err < 0) { fprintf(stderr, "bf16 forward failed\n"); return 1; }

    memcpy(bf16_last.data(), bf16_logits.data() + T_last * V, V * sizeof(float));

    // Compare last-token logits
    double max_diff = 0.0, sum_diff = 0.0, sum_sq = 0.0;
    double max_rel = 0.0;
    int n_out_of_tol = 0;
    double fp32_scale = 0.0;

    for (int i = 0; i < V; i++) {
        double a = fp32_last[i];
        double b = bf16_last[i];
        double diff = fabs(a - b);
        double rel = (fabs(a) > 1e-10) ? diff / fabs(a) : diff;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;
        sum_diff += diff;
        sum_sq += diff * diff;
        fp32_scale += a * a;
        if (diff > tolerance) n_out_of_tol++;
    }

    double rmse = sqrt(sum_sq / V);
    double rms_fp32 = sqrt(fp32_scale / V);
    double snr = (rms_fp32 > 1e-10) ? 20.0 * log10(rms_fp32 / rmse) : 0.0;

    // Argmax comparison
    int argmax_fp32 = 0, argmax_bf16 = 0;
    for (int i = 1; i < V; i++) {
        if (fp32_last[i] > fp32_last[argmax_fp32]) argmax_fp32 = i;
        if (bf16_last[i] > bf16_last[argmax_bf16]) argmax_bf16 = i;
    }

    fprintf(stderr, "\n=== bf16 vs fp32 alignment result ===\n");
    fprintf(stderr, "  Max absolute diff:  %e\n", max_diff);
    fprintf(stderr, "  Max relative diff:  %e\n", max_rel);
    fprintf(stderr, "  Mean absolute diff: %e\n", sum_diff / V);
    fprintf(stderr, "  RMSE:               %e\n", rmse);
    fprintf(stderr, "  Signal RMS (fp32):  %e\n", rms_fp32);
    fprintf(stderr, "  SNR:                %.2f dB\n", snr);
    fprintf(stderr, "  Args > tolerance (%.2e): %d / %d\n", tolerance, n_out_of_tol, V);
    fprintf(stderr, "  Argmax fp32: %d, bf16: %d %s\n",
        argmax_fp32, argmax_bf16,
        (argmax_fp32 == argmax_bf16) ? "✓ MATCH" : "✗ MISMATCH");

    // Top-5 comparison (manual sort, nvcc doesn't support std::partial_sort)
    std::vector<std::pair<float,int>> fp32_top(V), bf16_top(V);
    for (int i = 0; i < V; i++) {
        fp32_top[i] = {fp32_last[i], i};
        bf16_top[i] = {bf16_last[i], i};
    }
    auto top5_fn = [](std::vector<std::pair<float,int>>& vec) {
        for (int i = 0; i < 5 && i < (int)vec.size(); i++) {
            for (int j = i + 1; j < (int)vec.size(); j++) {
                if (vec[j].first > vec[i].first)
                    std::swap(vec[i], vec[j]);
            }
        }
    };
    top5_fn(fp32_top);
    top5_fn(bf16_top);

    fprintf(stderr, "  Top-5 fp32: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%d(%.2f) ", fp32_top[i].second, fp32_top[i].first);
    fprintf(stderr, "\n  Top-5 bf16: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%d(%.2f) ", bf16_top[i].second, bf16_top[i].first);
    fprintf(stderr, "\n");

    bool pass = (argmax_fp32 == argmax_bf16) && (max_diff < tolerance * 10);
    fprintf(stderr, "\n  === %s ===\n", pass ? "PASS" : "FAIL");

    w.free_all();
    bufs_fp32.free_all();
    bufs_bf16.free_all();
    cudaFree(d_ids);
    cudaStreamDestroy(s_fp32);
    cudaStreamDestroy(s_bf16);
    return pass ? 0 : 1;
}
