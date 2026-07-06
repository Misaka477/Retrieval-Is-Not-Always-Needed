// model_infer.cu — clean inference path (matches align_llama.cu approach)
#include "model.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include <cstdio>
#include <memory>
#include <vector>

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*, const float*, int, int, float, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_linear_dispatch(const void*, QuantType, const float*, float*, int, int, int, cudaStream_t);
extern float* g_dequant_tmp;
extern size_t g_dequant_tmp_sz;

static std::vector<std::unique_ptr<Layer>> a_layers;
static BufferManager a_bufs;

static bool has_deepseek_mla(const ModelConfig& cfg) {
    for (auto& lc : cfg.layers) {
        if (lc.type == "deepseek_mla_dense" || lc.type == "deepseek_mla_moe") return true;
    }
    return false;
}

void model_forward_direct(ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream,
    int start_pos) {

    int n=B*T, d=cfg.dim, V=cfg.vocab_size;
    a_bufs.fwd.kv_cache_quant.pre_rope = cfg.use_pre_rope_k;
    if(a_bufs.fwd.h==nullptr){
        a_layers=build_layers(cfg,w);
        auto* w1=w.get("transformer.h.0.mlp.w1.weight");
        int ws=0,total=0;
        for(auto& l:a_layers){
            int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;
            total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
        }
        int hd=(w1&&w1->n_dim>=2)?w1->shape[0]:(d*4*2/3/256*256);
        int max_seq = cfg.max_seq_len > 0 ? std::min(cfg.max_seq_len, 128) : 128;
        int infer_n = std::min(B * T, 512);
        bool is_mla = has_deepseek_mla(cfg);

        if (is_mla) {
            a_bufs.alloc_fwd(infer_n, d, ws, hd, V, total);
            if (!a_bufs.fwd.h) { fprintf(stderr,"ERROR: alloc_fwd failed\n"); return; }
            int dq = cfg.head_dim + cfg.d_h_r;
            a_bufs.alloc_attn_scratch(B, max_seq, cfg.n_heads, cfg.head_dim, dq);
            a_bufs.alloc_mla_kv_cache(cfg.n_layers, max_seq, cfg.n_kv_heads,
                                      cfg.d_h_r > 0 ? cfg.d_h_r : 64,
                                      cfg.head_dim, cfg.head_dim);
            if (!a_bufs.fwd.mla_kv_cache.data) { fprintf(stderr,"ERROR: mla_kv_cache alloc failed\n"); return; }
            // Pre-allocate dequant_tmp for all GGML matmuls
            if (!g_dequant_tmp) {
                int max_tmp = std::max(hd * d, d * hd) * (int)sizeof(float);
                cudaMalloc(&g_dequant_tmp, max_tmp);
                g_dequant_tmp_sz = g_dequant_tmp ? max_tmp : 0;
            }
        } else {
            a_bufs.alloc_fwd(infer_n, d, ws, hd, V, total);
            if (!g_dequant_tmp) {
                int max_tmp = std::max(hd * d, d * hd) * (int)sizeof(float);
                cudaMalloc(&g_dequant_tmp, max_tmp);
                g_dequant_tmp_sz = g_dequant_tmp ? max_tmp : 0;
            }
            if (cfg.kv_quant_mode != "fp32") {
                int mode = 5;
                if (cfg.kv_quant_mode == "q8") mode = 1;
                else if (cfg.kv_quant_mode == "q4") mode = 2;
                else if (cfg.kv_quant_mode == "q4k_q2v") mode = 3;
                else if (cfg.kv_quant_mode == "q2") mode = 4;
                else if (cfg.kv_quant_mode == "q2k_q1v") mode = 5;
                a_bufs.alloc_kv_cache_quant(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim, mode);
                a_bufs.alloc_kv_cache(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
            } else {
                a_bufs.alloc_kv_cache(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
            }
            a_bufs.alloc_attn_scratch(B, max_seq, cfg.n_heads, cfg.head_dim);
        }
    }

    a_bufs.fwd.kv_cache.start_pos = start_pos;
    a_bufs.fwd.kv_cache_quant.start_pos = start_pos;
    a_bufs.fwd.mla_kv_cache.start_pos = start_pos;

    if (a_layers.empty()) { fprintf(stderr,"ERROR: no layers built\n"); return; }
    auto* wte_t = w.get("transformer.wte.weight");
    if (!wte_t) { fprintf(stderr,"ERROR: wte not found\n"); return; }
    const float*wte=(const float*)wte_t->data;
    launch_embedding_fp32(wte,ids,a_bufs.fwd.h,B,T,d,stream);
    float*base_save=a_bufs.fwd.save;
    for(int l=0;l<(int)a_layers.size();l++){
        a_bufs.fwd.save=base_save+a_layers[l]->save_offset*n;
        a_layers[l]->forward(a_bufs.fwd.h,a_bufs.fwd,B,T,stream);
    }
    a_bufs.fwd.save=base_save;
    auto*ln_f_w=w.get("transformer.ln_f.weight");
    if(ln_f_w){
        launch_rms_norm_fp32(a_bufs.fwd.h,(const float*)ln_f_w->data,n,d,1e-5f,stream);
    }
    auto*lm_t=w.get("lm_head.weight");
    if(lm_t && lm_t->quant_type != QuantType::GGML_Q6_K) {
        launch_linear_dispatch(lm_t->data,lm_t->quant_type,a_bufs.fwd.h,a_bufs.fwd.lm,n,V,d,stream);
    } else {
        launch_linear_fp32(a_bufs.fwd.h,wte,a_bufs.fwd.lm,n,V,d,stream);
    }
    cudaError_t ce = cudaMemcpyAsync(logits,a_bufs.fwd.lm,n*V*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    if(ce != cudaSuccess) fprintf(stderr,"memcpy error: %s\n",cudaGetErrorString(ce));
}
