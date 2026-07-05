// model_forward_v2.cu — v2 forward path (uses ModelContext)
#include "model.h"
#include "model_v2_ctx.h"
#include <cstdio>

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_linear_dispatch(const void*,QuantType,const float*,float*,int,int,int,cudaStream_t);

ModelContext g_ctx;

void model_forward_v2(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream,
    int start_pos) {
    (void)start_pos;
    int n = B * T, d = cfg.dim, V = cfg.vocab_size;
    auto* w1_0 = w.get("transformer.h.0.mlp.w1.weight");
    int hd = (w1_0 && w1_0->n_dim >= 2) ? w1_0->shape[0] : (d * 4 * 2 / 3 / 256 * 256);

    if (!g_ctx.init(cfg, w)) return;
    int ws = compute_workspace_per_token(cfg, g_ctx.layers);
    g_ctx.ensure_bufs(n, d, ws, hd, V, false);

    auto& bufs = g_ctx.bufs;
    float* h = bufs.fwd.h;
    float* base_save = bufs.fwd.save;

    const float* wte = (const float*)w.get("transformer.wte.weight")->data;
    if (!wte) { fprintf(stderr, "v2: wte null\n"); return; }
    launch_embedding_fp32(wte, ids, h, B, T, d, stream);

    for (int l = 0; l < (int)g_ctx.layers.size(); l++) {
        bufs.fwd.save = base_save + g_ctx.layers[l]->save_offset * n;
        g_ctx.layers[l]->forward(h, bufs.fwd, B, T, stream);
    }
    bufs.fwd.save = base_save;

    {auto* ln_f_w = w.get("transformer.ln_f.weight");
    if (ln_f_w) {
        if (cfg.name.find("llama")==0)
            launch_rms_norm_fp32(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);
        else
            launch_pytorch_ln_kernel(h, (const float*)ln_f_w->data, n, d, 1e-5f, stream);
    }}

    auto* lm_t = w.get("lm_head.weight");
    if (lm_t) {
        launch_linear_dispatch(lm_t->data, lm_t->quant_type, h, bufs.fwd.lm, n, V, d, stream);
    } else {
        launch_linear_fp32(h, wte, bufs.fwd.lm, n, V, d, stream);
    }

    cudaMemcpyAsync(logits, bufs.fwd.lm, n * V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}
