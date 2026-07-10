#include "gguf_rina_bridge.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <fstream>

struct GGUFTensorRaw {
    std::string name;
    int ggml_type;
    int64_t shape[4];
    int n_dims;
    uint64_t file_offset;
};

static bool read_tensor_info_at(FILE * f, uint64_t info_offset, uint64_t idx,
                                GGUFTensorRaw & out) {
    fseek(f, (long)info_offset, SEEK_SET);
    for (uint64_t i = 0; i <= idx; i++) {
        uint64_t name_len; if (fread(&name_len, 8, 1, f) != 1) return false;
        std::string name((size_t)name_len, '\0');
        if (fread(&name[0], name_len, 1, f) != 1) return false;
        uint32_t n_dims; if (fread(&n_dims, 4, 1, f) != 1) return false;
        int64_t raw_dims[4] = {0,0,0,0};
        for (uint32_t rd = 0; rd < n_dims; rd++) {
            int64_t val; if (fread(&val, 8, 1, f) != 1) return false;
            raw_dims[rd] = val;
        }
        uint32_t dtype; if (fread(&dtype, 4, 1, f) != 1) return false;
        uint64_t foff; if (fread(&foff, 8, 1, f) != 1) return false;
        if (i == idx) {
            out.name = std::move(name);
            out.ggml_type = (int)dtype;
            out.n_dims = (int)n_dims;
            for (int d = 0; d < 4; d++) out.shape[d] = raw_dims[d];
            out.file_offset = foff;
            return true;
        }
    }
    return false;
}

static uint64_t read_gguf_header(FILE * f,
    std::unordered_map<std::string,std::string> & kv,
    uint64_t & n_tensors_out, uint64_t & data_start_out) {
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != 0x46554747) return 0;
    uint32_t version; fread(&version, 4, 1, f);
    (void)version;
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t key_len; fread(&key_len, 8, 1, f);
        std::string k((size_t)key_len, '\0');
        fread(&k[0], key_len, 1, f);
        uint32_t val_type; fread(&val_type, 4, 1, f);
        std::string val_str;
        switch (val_type) {
            case 0: { uint8_t v; fread(&v, 1, 1, f); val_str = std::to_string(v); break; }
            case 1: { int8_t v; fread(&v, 1, 1, f); val_str = std::to_string(v); break; }
            case 2: { uint16_t v; fread(&v, 2, 1, f); val_str = std::to_string(v); break; }
            case 3: { int16_t v; fread(&v, 2, 1, f); val_str = std::to_string(v); break; }
            case 4: { uint32_t v; fread(&v, 4, 1, f); val_str = std::to_string(v); break; }
            case 5: { int32_t v; fread(&v, 4, 1, f); val_str = std::to_string(v); break; }
            case 6: { float v; fread(&v, 4, 1, f); char buf[64]; snprintf(buf,64,"%f",v); val_str=buf; break; }
            case 7: { uint8_t v; fread(&v, 1, 1, f); val_str = v ? "true" : "false"; break; }
            case 10: { uint64_t v; fread(&v, 8, 1, f); val_str = std::to_string(v); break; }
            case 11: { int64_t v; fread(&v, 8, 1, f); val_str = std::to_string(v); break; }
            case 12: { double v; fread(&v, 8, 1, f); char buf[64]; snprintf(buf,64,"%f",v); val_str=buf; break; }
            case 8: {
                uint64_t len; fread(&len, 8, 1, f);
                val_str.resize((size_t)len);
                fread(&val_str[0], len, 1, f);
                break;
            }
            case 9: {
                uint32_t arr_type; fread(&arr_type, 4, 1, f);
                uint64_t arr_len; fread(&arr_len, 8, 1, f);
                int elem_size = 4;
                if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_size = 1;
                else if (arr_type == 2 || arr_type == 3) elem_size = 2;
                else if (arr_type == 4 || arr_type == 5 || arr_type == 6) elem_size = 4;
                else if (arr_type == 8) elem_size = -1;
                else if (arr_type == 11 || arr_type == 12) elem_size = 8;
                for (uint64_t j = 0; j < arr_len; j++) {
                    if (elem_size < 0) { uint64_t slen; fread(&slen, 8, 1, f); fseek(f, (long)slen, SEEK_CUR); }
                    else { fseek(f, elem_size, SEEK_CUR); }
                }
                val_str = "[array]";
                break;
            }
            default: break;
        }
        kv[k] = val_str;
    }
    uint64_t info_offset = ftell(f);
    n_tensors_out = n_tensors;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len; fread(&name_len, 8, 1, f);
        fseek(f, (long)name_len, SEEK_CUR);
        uint32_t n_dim; fread(&n_dim, 4, 1, f);
        fseek(f, (long)n_dim * 8, SEEK_CUR);
        fseek(f, 4 + 8, SEEK_CUR);
    }
    uint64_t end_offset = ftell(f);
    uint32_t align = 32;
    data_start_out = (end_offset + align - 1) & ~(align - 1);
    return info_offset;
}

static bool is_relevant_weight(const std::string & local) {
    return local == "token_embd.weight" || local == "output_norm.weight" || local == "output.weight" ||
           local == "attn_norm.weight" || local == "ffn_norm.weight" ||
           local == "attn_q.weight" || local == "attn_k.weight" || local == "attn_v.weight" ||
           local == "attn_output.weight" ||
           local == "ffn_gate.weight" || local == "ffn_up.weight" || local == "ffn_down.weight";
}

static enum ggml_type bridge_storage_type(const std::string & name, enum ggml_type file_type) {
    if (name == "token_embd.weight" && ggml_is_quantized(file_type)) {
        return GGML_TYPE_F32;
    }
    if (name == "output.weight.tied" && ggml_is_quantized(file_type)) {
        return file_type;
    }
    return file_type;
}

static size_t gguf_raw_tensor_size(const GGUFTensorRaw & ti) {
    int64_t n_elems = 1;
    for (int d = 0; d < ti.n_dims; d++) n_elems *= ti.shape[d];
    int blk_sz = ggml_blck_size((enum ggml_type)ti.ggml_type);
    int type_sz = ggml_type_size((enum ggml_type)ti.ggml_type);
    return (size_t)((n_elems + blk_sz - 1) / blk_sz) * type_sz;
}

BridgeModel::~BridgeModel() {
    if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
    if (weight_ctx) ggml_free(weight_ctx);
    if (backend) ggml_backend_free(backend);
    if (cpu_backend) ggml_backend_free(cpu_backend);
}

bool bridge_load_model(const char * path, BridgeModel & model) {
    std::unordered_map<std::string,std::string> kv;
    uint64_t n_tensors = 0, data_start = 0;
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    uint64_t info_offset = read_gguf_header(f, kv, n_tensors, data_start);
    if (!info_offset) { fprintf(stderr, "Bad GGUF header\n"); fclose(f); return false; }
    auto arch_it = kv.find("general.architecture");
    if (arch_it == kv.end()) { fprintf(stderr, "No architecture\n"); fclose(f); return false; }
    std::string arch = arch_it->second;
    BridgeConfig & cfg = model.config;
    auto gi = [&](const std::string & key, int def) -> int {
        auto it = kv.find(key); return it != kv.end() ? atoi(it->second.c_str()) : def;
    };
    auto gf = [&](const std::string & key, float def) -> float {
        auto it = kv.find(key); return it != kv.end() ? atof(it->second.c_str()) : def;
    };
    cfg.n_layers     = gi(arch + ".block_count", 0);
    cfg.dim          = gi(arch + ".embedding_length", 0);
    cfg.n_heads      = gi(arch + ".attention.head_count", 0);
    cfg.n_kv_heads   = gi(arch + ".attention.head_count_kv", cfg.n_heads);
    cfg.head_dim     = gi(arch + ".rope.dimension_count", 0);
    cfg.vocab_size   = gi(arch + ".vocab_size", 0);
    cfg.max_seq_len  = gi(arch + ".context_length", 2048);
    cfg.rope_freq_base      = gf(arch + ".rope.freq_base", 10000.0f);
    cfg.rope_scaling_factor = gf(arch + ".rope.frequency_scale", 1.0f);
    cfg.rms_norm_eps        = gf(arch + ".attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);
    if (cfg.head_dim == 0 && cfg.n_heads > 0 && cfg.dim > 0)
        cfg.head_dim = cfg.dim / cfg.n_heads;
    if (cfg.n_layers == 0 || cfg.dim == 0 || cfg.vocab_size == 0) {
        fprintf(stderr, "Incomplete config: layers=%d dim=%d vocab=%d\n",
                cfg.n_layers, cfg.dim, cfg.vocab_size);
        fclose(f); return false;
    }
    size_t total_weight_bytes = 0;
    int n_relevant = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorRaw ti;
        if (!read_tensor_info_at(f, info_offset, i, ti)) { fclose(f); return false; }
        std::string local = (ti.name.rfind("blk.", 0) == 0)
            ? ti.name.substr(ti.name.find('.', 4) + 1) : ti.name;
        if (!is_relevant_weight(local)) continue;
        int64_t n_elems = 1;
        for (int d = 0; d < ti.n_dims; d++) n_elems *= ti.shape[d];
        int blk_sz = ggml_blck_size((enum ggml_type)ti.ggml_type);
        int type_sz = ggml_type_size((enum ggml_type)ti.ggml_type);
        total_weight_bytes += (size_t)((n_elems + blk_sz - 1) / blk_sz) * type_sz;
        n_relevant++;
    }
    model.backend = ggml_backend_cuda_init(0);
    if (!model.backend) { fclose(f); fprintf(stderr, "ggml CUDA backend init failed\n"); return false; }
    model.cpu_backend = ggml_backend_cpu_init();
    if (!model.cpu_backend) { fclose(f); fprintf(stderr, "ggml CPU backend init failed\n"); return false; }
    model.buft = ggml_backend_cuda_buffer_type(0);

    size_t mem_size = ggml_tensor_overhead() * (n_relevant + 16);
    ggml_init_params params = {};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    model.weight_ctx = ggml_init(params);
    if (!model.weight_ctx) { fclose(f); fprintf(stderr, "ggml_init failed\n"); return false; }
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorRaw ti;
        if (!read_tensor_info_at(f, info_offset, i, ti)) continue;
        std::string local = (ti.name.rfind("blk.", 0) == 0)
            ? ti.name.substr(ti.name.find('.', 4) + 1) : ti.name;
        if (!is_relevant_weight(local)) continue;
        enum ggml_type wtype = bridge_storage_type(ti.name, (enum ggml_type)ti.ggml_type);
        int64_t ne0 = ti.shape[0];
        int64_t ne1 = ti.n_dims > 1 ? ti.shape[1] : 1;
        int64_t ne2 = ti.n_dims > 2 ? ti.shape[2] : 1;
        ggml_tensor * t = nullptr;
        if (ti.n_dims <= 1) t = ggml_new_tensor_1d(model.weight_ctx, wtype, ne0);
        else if (ti.n_dims == 2) t = ggml_new_tensor_2d(model.weight_ctx, wtype, ne0, ne1);
        else t = ggml_new_tensor_3d(model.weight_ctx, wtype, ne0, ne1, ne2);
        if (!t) { fprintf(stderr, "  tensor failed: %s\n", ti.name.c_str()); continue; }
        model.tensors[ti.name] = t;
        if (ti.name == "token_embd.weight" && wtype != (enum ggml_type)ti.ggml_type) {
            ggml_tensor * tied = nullptr;
            enum ggml_type file_type = (enum ggml_type)ti.ggml_type;
            if (ti.n_dims <= 1) tied = ggml_new_tensor_1d(model.weight_ctx, file_type, ne0);
            else if (ti.n_dims == 2) tied = ggml_new_tensor_2d(model.weight_ctx, file_type, ne0, ne1);
            else tied = ggml_new_tensor_3d(model.weight_ctx, file_type, ne0, ne1, ne2);
            if (!tied) { fprintf(stderr, "  tied tensor failed: %s\n", ti.name.c_str()); continue; }
            model.tensors["output.weight.tied"] = tied;
        }
    }
    model.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.weight_ctx, model.buft);
    if (!model.weight_buffer) { fclose(f); fprintf(stderr, "CUDA weight buffer allocation failed\n"); return false; }
    ggml_backend_buffer_set_usage(model.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorRaw ti;
        if (!read_tensor_info_at(f, info_offset, i, ti)) continue;
        std::string local = (ti.name.rfind("blk.", 0) == 0)
            ? ti.name.substr(ti.name.find('.', 4) + 1) : ti.name;
        if (!is_relevant_weight(local)) continue;
        auto it = model.tensors.find(ti.name);
        if (it == model.tensors.end()) continue;
        ggml_tensor * t = it->second;
        size_t raw_size = gguf_raw_tensor_size(ti);
        uint64_t abs_offset = data_start + ti.file_offset;
        std::vector<uint8_t> raw(raw_size);
        fseek(f, (long)abs_offset, SEEK_SET);
        fread(raw.data(), 1, raw_size, f);
        enum ggml_type file_type = (enum ggml_type)ti.ggml_type;
        if (t->type != file_type) {
            const int64_t n_elems = ggml_nelements(t);
            const ggml_type_traits * traits = ggml_get_type_traits(file_type);
            if (!traits || !traits->to_float) {
                fprintf(stderr, "cannot dequantize tensor: %s\n", ti.name.c_str());
                fclose(f); return false;
            }
            std::vector<float> tmp_f32((size_t)n_elems);
            traits->to_float(raw.data(), tmp_f32.data(), n_elems);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, tmp_f32.data(), 0, tmp_f32.size() * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp_f16((size_t)n_elems);
                ggml_fp32_to_fp16_row(tmp_f32.data(), tmp_f16.data(), n_elems);
                ggml_backend_tensor_set(t, tmp_f16.data(), 0, tmp_f16.size() * sizeof(ggml_fp16_t));
            } else {
                fprintf(stderr, "unsupported bridge storage type for tensor: %s\n", ti.name.c_str());
                fclose(f); return false;
            }
        } else {
            ggml_backend_tensor_set(t, raw.data(), 0, raw_size);
        }
        if (ti.name == "token_embd.weight") {
            auto tied_it = model.tensors.find("output.weight.tied");
            if (tied_it != model.tensors.end()) {
                ggml_backend_tensor_set(tied_it->second, raw.data(), 0, raw_size);
            }
        }
    }
    ggml_backend_synchronize(model.backend);
    fclose(f);
    fprintf(stderr, "  bridge CUDA: %s %dL %ddim %dH %dhead %dV (%d tensors, %zu MB)\n",
            arch.c_str(), cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim,
            cfg.vocab_size, n_relevant, total_weight_bytes / 1048576);
    return true;
}

static ggml_tensor * get_w(const std::map<std::string,ggml_tensor*> & tensors, const std::string & name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) fprintf(stderr, "  missing: %s\n", name.c_str());
    return it != tensors.end() ? it->second : nullptr;
}
static ggml_tensor * get_lw(const std::map<std::string,ggml_tensor*> & tensors, int l, const std::string & local) {
    return get_w(tensors, "blk." + std::to_string(l) + "." + local);
}

float * bridge_forward(BridgeModel & model, const int32_t * tokens, int n_tokens) {
    BridgeConfig & cfg = model.config;
    size_t graph_mem = 64ull * 1024 * 1024;
    ggml_init_params params = {};
    params.mem_size = graph_mem;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) { fprintf(stderr, "ggml_init graph failed\n"); return nullptr; }

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_input(kq_mask);

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++) pos[i] = i;
    std::vector<float> mask((size_t)n_tokens * n_tokens);
    for (int i = 0; i < n_tokens; i++)
        for (int j = 0; j < n_tokens; j++)
            mask[(size_t)i * n_tokens + j] = (j <= i) ? 0.0f : -INFINITY;

    ggml_tensor * tok_embd = get_w(model.tensors, "token_embd.weight");
    if (!tok_embd) { ggml_free(ctx0); return nullptr; }
    ggml_tensor * cur = ggml_get_rows(ctx0, tok_embd, inp_tokens);

    int n_rot = cfg.head_dim;
    float freq_base = cfg.rope_freq_base;
    int n_ctx_orig = cfg.max_seq_len;

    for (int il = 0; il < cfg.n_layers; il++) {
        ggml_tensor * an_w = get_lw(model.tensors, il, "attn_norm.weight");
        ggml_tensor * wq  = get_lw(model.tensors, il, "attn_q.weight");
        ggml_tensor * wk  = get_lw(model.tensors, il, "attn_k.weight");
        ggml_tensor * wv  = get_lw(model.tensors, il, "attn_v.weight");
        ggml_tensor * wo  = get_lw(model.tensors, il, "attn_output.weight");
        if (!an_w || !wq || !wk || !wv || !wo) { ggml_free(ctx0); return nullptr; }

        ggml_tensor * normed = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        normed = ggml_mul(ctx0, normed, an_w);
        ggml_tensor * Q = ggml_mul_mat(ctx0, wq, normed);
        ggml_tensor * K = ggml_mul_mat(ctx0, wk, normed);
        ggml_tensor * V = ggml_mul_mat(ctx0, wv, normed);
        Q = ggml_reshape_3d(ctx0, Q, cfg.head_dim, cfg.n_heads, n_tokens);
        K = ggml_reshape_3d(ctx0, K, cfg.head_dim, cfg.n_kv_heads, n_tokens);
        V = ggml_reshape_3d(ctx0, V, cfg.head_dim, cfg.n_kv_heads, n_tokens);
        Q = ggml_rope_ext(ctx0, Q, inp_pos, nullptr,
                          n_rot, 0, n_ctx_orig, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, inp_pos, nullptr,
                          n_rot, 0, n_ctx_orig, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const bool v_trans = V->nb[1] > V->nb[2];
        Q = ggml_reshape_4d(ctx0, Q, Q->ne[0], Q->ne[1], Q->ne[2], 1);
        K = ggml_reshape_4d(ctx0, K, K->ne[0], K->ne[1], K->ne[2], 1);
        V = ggml_reshape_4d(ctx0, V, V->ne[0], V->ne[1], V->ne[2], 1);
        ggml_tensor * Qp = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        ggml_tensor * Kp = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor * Vp = ggml_permute(ctx0, V, 0, 2, 1, 3);
        float scale = 1.0f / sqrtf((float)cfg.head_dim);
        ggml_tensor * kq = ggml_mul_mat(ctx0, Kp, Qp);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, scale, 0.0f);
        if (!v_trans) {
            Vp = ggml_cont(ctx0, ggml_transpose(ctx0, Vp));
        }
        ggml_tensor * kqv = ggml_mul_mat(ctx0, Vp, kq);
        ggml_tensor * attn = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx0, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, wo, attn), cur);

        ggml_tensor * fn_w = get_lw(model.tensors, il, "ffn_norm.weight");
        ggml_tensor * w1  = get_lw(model.tensors, il, "ffn_gate.weight");
        ggml_tensor * w2  = get_lw(model.tensors, il, "ffn_down.weight");
        ggml_tensor * w3  = get_lw(model.tensors, il, "ffn_up.weight");
        if (!fn_w || !w1 || !w2 || !w3) { ggml_free(ctx0); return nullptr; }
        normed = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        normed = ggml_mul(ctx0, normed, fn_w);
        cur = ggml_add(ctx0,
            ggml_mul_mat(ctx0, w2, ggml_mul(ctx0,
                ggml_silu(ctx0, ggml_mul_mat(ctx0, w1, normed)),
                ggml_mul_mat(ctx0, w3, normed))),
            cur);
    }

    ggml_tensor * out_norm = get_w(model.tensors, "output_norm.weight");
    auto out_it = model.tensors.find("output.weight");
    auto tied_out_it = model.tensors.find("output.weight.tied");
    ggml_tensor * out_w = out_it != model.tensors.end() ? out_it->second :
        (tied_out_it != model.tensors.end() ? tied_out_it->second : get_w(model.tensors, "token_embd.weight"));
    if (!out_norm || !out_w) { ggml_free(ctx0); return nullptr; }
    cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
    cur = ggml_mul(ctx0, cur, out_norm);
    ggml_tensor * logits = ggml_mul_mat(ctx0, out_w, cur);
    ggml_set_output(logits);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_t backends[] = { model.backend, model.cpu_backend };
    ggml_backend_buffer_type_t bufts[] = { model.buft, ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { ggml_free(ctx0); return nullptr; }
    ggml_backend_sched_set_tensor_backend(sched, inp_tokens, model.backend);
    ggml_backend_sched_set_tensor_backend(sched, inp_pos, model.backend);
    ggml_backend_sched_set_tensor_backend(sched, kq_mask, model.backend);
    ggml_backend_sched_set_tensor_backend(sched, logits, model.backend);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "graph backend allocation failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_set(inp_tokens, tokens, 0, (size_t)n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, (size_t)n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %d\n", (int)status);
        ggml_backend_sched_free(sched);
        ggml_free(ctx0);
        return nullptr;
    }

    size_t logits_size = (size_t)cfg.vocab_size * n_tokens * sizeof(float);
    float * result = (float *)malloc(logits_size);
    if (result) {
        ggml_backend_tensor_get(logits, result, 0, logits_size);
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx0);
    return result;
}
