// model_infer.cu — clean inference path (matches align_llama.cu approach)
#include "model.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include <cstdio>
#include <memory>
#include <vector>

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);

static std::vector<std::unique_ptr<Layer>> a_layers;
static BufferManager a_bufs;

void model_forward_direct(const ModelConfig& cfg, const TensorMap& w,
    const int* ids, float* logits, int B, int T, cudaStream_t stream) {
    int n=B*T, d=cfg.dim, V=cfg.vocab_size;
    if(a_bufs.fwd.h==nullptr){
        a_layers=build_layers(cfg,w);
        auto* w1=w.get("transformer.h.0.mlp.w1.weight");
        int ws=0,total=0;
        for(auto& l:a_layers){
            int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;
            total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
        }
        int hd=(w1&&w1->n_dim>=2)?w1->shape[0]:(d*4*2/3/256*256);
        a_bufs.alloc_fwd(512,d,ws,hd,V,total);
    }
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte,ids,a_bufs.fwd.h,B,T,d,stream);
    float*base_save=a_bufs.fwd.save;
    for(int l=0;l<(int)a_layers.size();l++){
        a_bufs.fwd.save=base_save+a_layers[l]->save_offset*n;
        a_layers[l]->forward(a_bufs.fwd.h,a_bufs.fwd,B,T,stream);
    }
    a_bufs.fwd.save=base_save;
    auto*ln_f_w=w.get("transformer.ln_f.weight");
    if(ln_f_w){
        // Llama uses RMSNorm; our custom models (rina-*) use LayerNorm
        if(cfg.name.find("llama")==0)
            launch_rms_norm_fp32(a_bufs.fwd.h,(const float*)ln_f_w->data,n,d,1e-5f,stream);
        else
            launch_pytorch_ln_kernel(a_bufs.fwd.h,(const float*)ln_f_w->data,n,d,1e-5f,stream);
    }
    auto*lm_t=w.get("lm_head.weight");
    const float*lm_w=lm_t?(const float*)lm_t->data:wte;
    launch_linear_fp32(a_bufs.fwd.h,lm_w,a_bufs.fwd.lm,n,V,d,stream);
    cudaMemcpyAsync(logits,a_bufs.fwd.lm,n*V*sizeof(float),cudaMemcpyDeviceToDevice,stream);
}
