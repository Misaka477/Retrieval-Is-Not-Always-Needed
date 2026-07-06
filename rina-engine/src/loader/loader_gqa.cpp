#include "loader/loader_base.h"
#include <cstdio>
#include <string>

static const TensorNameEntry gqa_names[] = {
    {"token_embd.weight",    "transformer.wte.weight"},
    {"output_norm.weight",   "transformer.ln_f.weight"},
    {"output.weight",        "lm_head.weight"},
};

static const TensorNameEntry gqa_layer_names[] = {
    {"attn_norm.weight",     "ln1.weight"},
    {"ffn_norm.weight",      "ln2.weight"},
    {"attn_q.weight",        "attn.w_q.weight"},
    {"attn_k.weight",        "attn.w_k.weight"},
    {"attn_v.weight",        "attn.w_v.weight"},
    {"attn_output.weight",   "attn.w_o.weight"},
    {"ffn_gate.weight",      "mlp.w1.weight"},
    {"ffn_up.weight",        "mlp.w3.weight"},
    {"ffn_down.weight",      "mlp.w2.weight"},
};

static bool detect_gqa(const GGUFMetadata& meta) {
    return meta.arch_string == "llama" || meta.d_c == 0 || meta.dq == 0;
}

static bool map_names_gqa(const GGUFMetadata& meta,
                          std::vector<TensorNameEntry>& out_names) {
    out_names.clear();
    for (auto& e : gqa_names)
        out_names.push_back(e);
    for (int l = 0; l < meta.n_layers; l++) {
        for (auto& e : gqa_layer_names) {
            std::string gguf = "blk." + std::to_string(l) + "." + e.gguf_name;
            std::string rinn = "transformer.h." + std::to_string(l) + "." + e.rinn_name;
            out_names.push_back({strdup(gguf.c_str()), strdup(rinn.c_str())});
        }
    }
    return true;
}

static bool build_config_gqa(ModelConfig& cfg, const GGUFMetadata& meta) {
    int head_dim = meta.dim / meta.n_heads;
    int n_kv_heads = meta.n_kv_heads > 0 ? meta.n_kv_heads : meta.n_heads;
    int max_seq_len = meta.max_seq_len > 0 ? meta.max_seq_len : 2048;

    cfg.name = "gguf-llama-" + std::to_string(meta.dim / 1000) + "dim";
    cfg.dim = meta.dim;
    cfg.n_layers = meta.n_layers;
    cfg.n_heads = meta.n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.head_dim = head_dim;
    cfg.d_c = 0;
    cfg.d_h_r = meta.d_h_r > 0 ? meta.d_h_r : head_dim;
    cfg.vocab_size = meta.vocab_size;
    cfg.max_seq_len = std::min(max_seq_len, 1024);
    cfg.ssm_steps = 0;
    cfg.weight_tying = true;
    cfg.layers.clear();
    for (int i = 0; i < meta.n_layers; i++)
        cfg.layers.push_back({"standard_attention", "layer_" + std::to_string(i), 1, {}});
    return true;
}

static ArchLoader gqa_loader = {
    "gqa",
    detect_gqa,
    map_names_gqa,
    build_config_gqa,
};

// Static initializer to register
static int gqa_registered = (register_loader(gqa_loader), 0);
