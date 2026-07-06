#include "infer/infer_base.h"
#include "core/layer.h"
#include "core/quant.h"
#include "training/train.h"
#include <cstdio>
#include <memory>
#include <vector>

#include "ops/embedding.h"
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
extern void launch_linear_dispatch(const void*, QuantType, const float*, float*, int, int, int, cudaStream_t);

extern float* g_dequant_tmp;
extern size_t g_dequant_tmp_sz;

struct MLAInference : Inference {
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
        int dq = cfg_.head_dim + cfg_.d_h_r;

        bufs.alloc_fwd(infer_n, d, ws, hd, V, total);
        if (!bufs.fwd.h) { fprintf(stderr,"ERROR: alloc_fwd failed\n"); return false; }

        bufs.alloc_attn_scratch(1, max_seq, cfg_.n_heads, cfg_.head_dim, dq);
        bufs.alloc_mla_kv_cache(cfg_.n_layers, max_seq, cfg_.n_kv_heads,
                                cfg_.d_h_r > 0 ? cfg_.d_h_r : 64,
                                cfg_.head_dim, cfg_.head_dim);
        if (!bufs.fwd.mla_kv_cache.data) { fprintf(stderr,"ERROR: mla_kv_cache alloc failed\n"); return false; }

        if (!g_dequant_tmp) {
            int max_tmp = hd * d;
            if (d * hd > max_tmp) max_tmp = d * hd;
            max_tmp *= (int)sizeof(float);
            cudaMalloc(&g_dequant_tmp, max_tmp);
            g_dequant_tmp_sz = g_dequant_tmp ? max_tmp : 0;
        }

        wte = weights.get("transformer.wte.weight");
        ln_f = weights.get("transformer.ln_f.weight");
        lm_head = weights.get("lm_head.weight");

        return !layers.empty() && bufs.fwd.h && wte;
    }

    void forward(const int* ids, float* logits,
                 int B, int T, int start_pos, cudaStream_t stream) override {
        int n = B * T, d = cfg.dim, V = cfg.vocab_size;

        bufs.fwd.mla_kv_cache.start_pos = start_pos;

        launch_embedding(wte->data, wte->quant_type, ids, bufs.fwd.h, B, T, d, stream);

        float* base_save = bufs.fwd.save;
        for (int l = 0; l < (int)layers.size(); l++) {
            bufs.fwd.save = base_save + layers[l]->save_offset * n;
            layers[l]->forward(bufs.fwd.h, bufs.fwd, B, T, stream);
        }
        bufs.fwd.save = base_save;

        if (ln_f) {
            launch_rms_norm_fp32(bufs.fwd.h, (const float*)ln_f->data, n, d, 1e-5f, stream);
        }

        // Use dedicated lm_head if available, else tied embedding
        if (lm_head) {
            // Tile lm_head in chunks to avoid large dequant temp buffer
            int chunk = 8192;     // process 8K rows at a time → ~64MB temp buffer
            for (int vo = 0; vo < V; vo += chunk) {
                int vc = std::min(chunk, V - vo);
                int blk_sz = ggml_block_size(lm_head->quant_type);
                int type_sz = ggml_type_size(lm_head->quant_type);
                int n_blk = (vc * d + blk_sz - 1) / blk_sz;
                int row_bytes = ((d + blk_sz - 1) / blk_sz) * type_sz;
                const void* chunk_data = (const uint8_t*)lm_head->data + (size_t)vo * row_bytes;
                launch_linear_dispatch(chunk_data, lm_head->quant_type,
                    bufs.fwd.h, bufs.fwd.lm + vo, n, vc, d, stream);
            }
        } else {
            launch_linear_dispatch(wte->data, wte->quant_type,
                bufs.fwd.h, bufs.fwd.lm, n, V, d, stream);
        }

        cudaMemcpyAsync(logits, bufs.fwd.lm, n * V * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
};

Inference* create_mla_inference() {
    return new MLAInference();
}
