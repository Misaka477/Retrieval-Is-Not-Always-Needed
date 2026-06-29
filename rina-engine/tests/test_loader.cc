#include "core/config.h"
#include "core/tensor.h"
#include <cstdio>
#include <cstring>

bool load_rinn(const char* path, ModelConfig& cfg, TensorMap& weights);

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/test_qw.rinn";

    printf("=== Test: Load .rinn ===\n");
    printf("File: %s\n", path);

    ModelConfig cfg;
    TensorMap weights;

    if (!load_rinn(path, cfg, weights)) {
        printf("FAIL: load_rinn\n");
        return 1;
    }

    printf("OK: config parsed\n");
    printf("  name=%s layers=%d dim=%d vocab=%d\n",
           cfg.name.c_str(), cfg.n_layers, cfg.dim, cfg.vocab_size);
    printf("  heads=%d kv_heads=%d d_c=%d\n",
           cfg.n_heads, cfg.n_kv_heads, cfg.d_c);

    // 验证关键 tensor 存在
    const char* checks[] = {
        "transformer.wte.weight",
        "transformer.ln_f.weight",
        "transformer.h.0.ln1.weight",
        "transformer.h.0.mlp.w1.weight",
        "transformer.h.0.path.w_dq.weight",
        "transformer.h.3.path.w_dqkv.weight",
        "lm_head.weight",
    };
    int ok = 0;
    for (auto* name : checks) {
        if (weights.get(name)) {
            printf("  %-50s ✅\n", name);
            ok++;
        } else {
            printf("  %-50s ❌\n", name);
        }
    }
    printf("Tensors found: %d/%zu\n", ok, sizeof(checks)/sizeof(checks[0]));

    weights.free_all();
    printf("=== All loader tests ===\n");
    return ok == (int)(sizeof(checks)/sizeof(checks[0])) ? 0 : 1;
}
