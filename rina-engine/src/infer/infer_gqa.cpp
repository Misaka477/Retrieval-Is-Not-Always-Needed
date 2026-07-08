#include "infer/infer_base.h"
#include "core/layer.h"
#include "training/train.h"
#include <cstdio>
#include <memory>
#include <vector>

#include "ops/embedding.h"
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_linear_dispatch(const void*, QuantType, const float*, float*, int, int, int, cudaStream_t);

extern float* g_dequant_tmp;
extern size_t g_dequant_tmp_sz;

struct GQAInference : Inference {
    std::vector<std::unique_ptr<Layer>> layers;
    BufferManager bufs;
    ModelConfig cfg;
    const WeightTensor* wte = nullptr;
    const WeightTensor* ln_f = nullptr;
    const WeightTensor* lm_head = nullptr;

    bool init(ModelConfig& cfg_, const TensorMap& weights) override {
        cfg = cfg_;
        int d = cfg.dim, V = cfg.vocab_size;

        layers = build_layers(cfg_, weights);
        auto* w1 = weights.get("transformer.h.0.mlp.w1.weight");
        int hd = (w1 && w1->n_dim >= 2) ? w1->shape[0] : (d * 4 * 2 / 3 / 256 * 256);
        int ws = 0, total = 0;
        for (auto& l : layers) {
            int w = l->workspace_per_token(d, cfg_.n_heads, cfg_.head_dim);
            if (w > ws) ws = w;
            total += l->saved_per_token(d, cfg_.n_heads, cfg_.head_dim);
        }

        int max_seq = cfg_.max_seq_len > 0 ? std::min(cfg_.max_seq_len, 128) : 128;
        int infer_n = std::min(512, max_seq);

        bufs.alloc_fwd(infer_n, d, ws, hd, V, total);

        if (!g_dequant_tmp) {
            int max_tmp = hd * d;
            if (d * hd > max_tmp) max_tmp = d * hd;
            if (V * d > max_tmp) max_tmp = V * d;
            max_tmp *= (int)sizeof(float);
            cudaMalloc(&g_dequant_tmp, max_tmp);
            g_dequant_tmp_sz = g_dequant_tmp ? max_tmp : 0;
            if (!g_dequant_tmp) fprintf(stderr, "  WARNING: g_dequant_tmp OOM (%d MB)\n", max_tmp / 1048576);
        }

        if (cfg_.kv_quant_mode != "fp32") {
            int mode = 5;
            if (cfg_.kv_quant_mode == "q8") mode = 1;
            else if (cfg_.kv_quant_mode == "q4") mode = 2;
            else if (cfg_.kv_quant_mode == "q4k_q2v") mode = 3;
            else if (cfg_.kv_quant_mode == "q2") mode = 4;
            else if (cfg_.kv_quant_mode == "q2k_q1v") mode = 5;
            bufs.alloc_kv_cache_quant(cfg_.n_layers, max_seq, cfg_.n_kv_heads, cfg_.head_dim, mode);
            bufs.alloc_kv_cache(cfg_.n_layers, max_seq, cfg_.n_kv_heads, cfg_.head_dim);
        } else {
            bufs.alloc_kv_cache(cfg_.n_layers, max_seq, cfg_.n_kv_heads, cfg_.head_dim);
        }
        bufs.alloc_attn_scratch(1, max_seq, cfg_.n_heads, cfg_.head_dim);

        wte = weights.get("transformer.wte.weight");
        ln_f = weights.get("transformer.ln_f.weight");
        lm_head = weights.get("lm_head.weight");

        return !layers.empty() && bufs.fwd.h && wte;
    }

    void forward(const int* ids, float* logits,
                 int B, int T, int start_pos, cudaStream_t stream) override {
        int n = B * T, d = cfg.dim, V = cfg.vocab_size;

        bufs.fwd.kv_cache.start_pos = start_pos;
        bufs.fwd.kv_cache_quant.start_pos = start_pos;

        launch_embedding(wte->data, wte->quant_type, ids, bufs.fwd.h, B, T, d, stream);

        float* base_save = bufs.fwd.save;
        for (int l = 0; l < (int)layers.size(); l++) {
            bufs.fwd.save = base_save + layers[l]->save_offset * n;
            layers[l]->forward(bufs.fwd.h, bufs.fwd, B, T, stream);
        }
        bufs.fwd.save = base_save;

        if (ln_f) {
            bool use_rms = (cfg.name.find("llama") != std::string::npos);
            if (use_rms) launch_rms_norm_fp32(bufs.fwd.h, (const float*)ln_f->data, n, d, 1e-5f, stream);
            else launch_pytorch_ln_kernel(bufs.fwd.h, (const float*)ln_f->data, n, d, 1e-5f, stream);
        }

        if (lm_head && lm_head->quant_type != QuantType::GGML_Q6_K) {
            launch_linear_dispatch(lm_head->data, lm_head->quant_type, bufs.fwd.h, bufs.fwd.lm, n, V, d, stream);
        } else if (wte) {
            launch_linear_dispatch(wte->data, wte->quant_type, bufs.fwd.h, bufs.fwd.lm, n, V, d, stream);
        }

        cudaMemcpyAsync(logits, bufs.fwd.lm, n * V * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
};

Inference* create_gqa_inference() {
    return new GQAInference();
}
