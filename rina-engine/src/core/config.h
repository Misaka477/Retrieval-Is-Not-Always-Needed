#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cstdlib>

struct LayerConfig {
    std::string type;          // "inertia_wave_ssm", "sparse_gather_fa", "swiglu", ...
    std::string name;          // e.g. "layer_0"
    int layer_type;            // 0=SSM, 1=Sparse (for Jamba)
    std::unordered_map<std::string, int> args;  // K, W, ssm_qbits, ...
};

struct TensorSpec {
    std::string name;
    int shape[4];
    int n_dim;
    uint8_t quant_type;   // QuantType enum value
    uint16_t block_size;
    uint64_t offset;
    uint64_t size;
};

struct ModelConfig {
    // model topology
    std::string name;
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int d_c;               // MLA latent dim
    int d_h_r;             // RoPE head dim
    int vocab_size;
    int max_seq_len;
    int ssm_steps;
    bool weight_tying;

    // quantization
    std::string quant_mode;  // "q2k_q1v", "q4k_q2v"
    int ssm_qbits;
    bool use_bf16 = false;
    std::string kv_quant_mode = "fp32";
    bool use_pre_rope_k = false;  // cache pre-RoPE K instead of post-RoPE

    // layers
    std::vector<LayerConfig> layers;

    // kv cache quantization
    std::string kv_k_quant;  // "q2_1"
    std::string kv_v_quant;  // "q1_0"

    // tensor index (populated by loader)
    std::vector<TensorSpec> tensors;
    std::unordered_map<std::string, int> tensor_map;

    int find_tensor(const std::string& name) const {
        auto it = tensor_map.find(name);
        return it != tensor_map.end() ? it->second : -1;
    }
};

// ——— 极简 JSON 解析（仅支持 config 所需子集） ———

static inline std::string json_trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static inline std::string json_strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

static inline int json_parse_int(const std::string& s) {
    return atoi(s.c_str());
}

// 从 JSON 字符串中查找 key 的值（仅基础类型，不支持嵌套）
// e.g. json_get_key(R"({"dim": 640, "name": "test"})", "dim") → "640"
static inline std::string json_get_key(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;  // skip ':'
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        if (end != std::string::npos) return json_trim(json.substr(pos, end - pos + 1));
    }
    // number or bool
    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n') end++;
    return json_trim(json.substr(pos, end - pos));
}

// JSON 数组解析
static inline std::vector<std::string> json_parse_array(const std::string& json) {
    std::vector<std::string> items;
    size_t start = json.find('[');
    size_t end   = json.find(']');
    if (start == std::string::npos || end == std::string::npos) return items;
    std::string content = json.substr(start + 1, end - start - 1);
    // split by '}' (object boundary)
    size_t pos = 0;
    while (true) {
        size_t obj_start = content.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end   = content.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        items.push_back(content.substr(obj_start, obj_end - obj_start + 1));
        pos = obj_end + 1;
    }
    return items;
}

// 解析 ModelConfig 从 JSON 字符串
static inline ModelConfig parse_config(const std::string& json) {
    ModelConfig cfg{};
    cfg.name        = json_strip_quotes(json_get_key(json, "name"));
    cfg.dim         = json_parse_int(json_get_key(json, "dim"));
    cfg.n_layers    = json_parse_int(json_get_key(json, "n_layers"));
    cfg.n_heads     = json_parse_int(json_get_key(json, "n_heads"));
    cfg.n_kv_heads  = json_parse_int(json_get_key(json, "n_kv_heads"));
    cfg.head_dim    = json_parse_int(json_get_key(json, "head_dim"));
    cfg.d_c         = json_parse_int(json_get_key(json, "d_c"));
    cfg.d_h_r       = json_parse_int(json_get_key(json, "d_h_r"));
    cfg.vocab_size  = json_parse_int(json_get_key(json, "vocab_size"));
    cfg.max_seq_len = json_parse_int(json_get_key(json, "max_seq_len"));
    cfg.ssm_steps   = json_parse_int(json_get_key(json, "ssm_steps"));
    cfg.weight_tying = json_get_key(json, "weight_tying") == "true";

    // layers array
    std::string layers_str = json_get_key(json, "layers");
    if (!layers_str.empty()) {
        // find the actual array
        size_t arr_start = json.find("\"layers\"");
        if (arr_start != std::string::npos) {
            arr_start = json.find('[', arr_start);
            size_t arr_end = json.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr_content = json.substr(arr_start, arr_end - arr_start + 1);
                auto items = json_parse_array(arr_content);
                for (int i = 0; i < (int)items.size(); i++) {
                    LayerConfig lc;
                    lc.type = json_strip_quotes(json_get_key(items[i], "type"));
                    lc.name = "layer_" + std::to_string(i);
                    lc.layer_type = (lc.type.find("ssm") != std::string::npos) ? 0 : 1;
                    std::string args_str = json_get_key(items[i], "args");
                    if (args_str.find("K") != std::string::npos)
                        lc.args["K"] = json_parse_int(json_get_key(items[i], "K"));
                    if (args_str.find("W") != std::string::npos)
                        lc.args["W"] = json_parse_int(json_get_key(items[i], "W"));
                    cfg.layers.push_back(lc);
                }
            }
        }
    }

    return cfg;
}
