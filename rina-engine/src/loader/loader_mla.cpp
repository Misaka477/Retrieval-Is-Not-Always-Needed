#include "loader/loader_base.h"
#include <cstdio>
#include <string>

static const TensorNameEntry mla_names[] = {
    {"token_embd.weight",    "transformer.wte.weight"},
    {"output.weight",        "lm_head.weight"},
    {"output_norm.weight",   "transformer.ln_f.weight"},
};

static const TensorNameEntry mla_layer_names[] = {
    {"attn_norm.weight",        "ln1.weight"},
    {"ffn_norm.weight",         "ln2.weight"},
    {"attn_kv_a_mqa.weight",    "attn.w_kv_a.weight"},
    {"attn_kv_a_norm.weight",   "attn.k_norm.weight"},
    {"attn_kv_b.weight",        "attn.w_kv_b.weight"},
    {"ffn_gate_inp.weight",     "mlp.gate_inp.weight"},
    {"ffn_gate_exps.weight",    "mlp.gate_exps.weight"},
    {"ffn_up_exps.weight",      "mlp.up_exps.weight"},
    {"ffn_down_exps.weight",    "mlp.down_exps.weight"},
    {"ffn_gate_shexp.weight",   "mlp.gate_shexp.weight"},
    {"ffn_up_shexp.weight",     "mlp.up_shexp.weight"},
    {"ffn_down_shexp.weight",   "mlp.down_shexp.weight"},
};

static bool detect_mla(const GGUFMetadata& meta) {
    return meta.d_c > 0 && meta.dq > 0;
}

static bool map_names_mla(const GGUFMetadata& meta,
                          std::vector<TensorNameEntry>& out_names) {
    out_names.clear();
    for (auto& e : mla_names)
        out_names.push_back(e);
    for (int l = 0; l < meta.n_layers; l++) {
        for (auto& e : mla_layer_names) {
            std::string gguf = "blk." + std::to_string(l) + "." + e.gguf_name;
            std::string rinn = "transformer.h." + std::to_string(l) + "." + e.rinn_name;
            out_names.push_back({strdup(gguf.c_str()), strdup(rinn.c_str())});
        }
    }
    return true;
}

static bool build_config_mla(ModelConfig& cfg, const GGUFMetadata& meta) {
    int n_kv_heads = meta.n_kv_heads > 0 ? meta.n_kv_heads : meta.n_heads;
    int max_seq_len = meta.max_seq_len > 0 ? meta.max_seq_len : 2048;
    // For DeepSeek MLA: value_length is the true head_dim, dimension_count is d_h_r (RoPE sub-dim)
    int head_dim = meta.v_head_dim > 0 ? meta.v_head_dim : (meta.dim / meta.n_heads);

    cfg.name = "deepseek2-" + std::to_string(meta.dim / 1000) + "dim";
    cfg.dim = meta.dim;
    cfg.n_layers = meta.n_layers;
    cfg.n_heads = meta.n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.head_dim = head_dim;
    cfg.d_c = meta.d_c;
    cfg.d_h_r = meta.d_h_r;
    cfg.vocab_size = meta.vocab_size;
    cfg.max_seq_len = std::min(max_seq_len, 1024);
    cfg.ssm_steps = 0;
    cfg.weight_tying = true;
    cfg.layers.clear();
    for (int i = 0; i < meta.n_layers; i++) {
        // Layer 0 is dense (leading_dense_block_count=1), rest are MoE
        cfg.layers.push_back({
            i == 0 ? "deepseek_mla_dense" : "deepseek_mla_moe",
            "layer_" + std::to_string(i), 1, {}});
    }
    return true;
}

static ArchLoader mla_loader = {
    "mla",
    detect_mla,
    map_names_mla,
    build_config_mla,
};

static int mla_registered = (register_loader(mla_loader), 0);
