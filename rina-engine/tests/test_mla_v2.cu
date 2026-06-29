#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/model_a_v2.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    fprintf(stderr,"Model: dim=%d layers=%d heads=%d kv=%d dc=%d dhr=%d V=%d\n",
        cfg.dim,cfg.n_layers,cfg.n_heads,cfg.n_kv_heads,cfg.d_c,cfg.d_h_r,cfg.vocab_size);

    auto layers=build_layers(cfg,w);
    if(layers.empty()){fprintf(stderr,"build layers fail\n");return 1;}
    fprintf(stderr,"%zu layers built\n",layers.size());

    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int hd=d*4*2/3/256*256;
    int ws=0,total_saved=0;
    for(auto&l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);
        if(w>ws)ws=w;
        total_saved+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    fprintf(stderr,"ws=%d saved=%d\n",ws,total_saved);

    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total_saved);
    if(!bufs.fwd.h||!bufs.fwd.a||!bufs.fwd.m||!bufs.fwd.save){fprintf(stderr,"alloc fail\n");return 1;}

    cudaStream_t s;cudaStreamCreate(&s);
    int*ids;float*logits;
    cudaMalloc(&ids,n*sizeof(int));
    cudaMalloc(&logits,n*V*sizeof(float));

    float*h=bufs.fwd.h;
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte,ids,h,B,T,d,s);
    cudaStreamSynchronize(s);
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){fprintf(stderr,"embed: %s\n",cudaGetErrorString(e));return 1;}
    fprintf(stderr,"embed OK\n");

    float* base_save=bufs.fwd.save;
    for(int l=0;l<(int)layers.size();l++){
        fprintf(stderr,"layer %d (save_off=%d)...\n",l,layers[l]->save_offset);
        bufs.fwd.save=base_save+layers[l]->save_offset*n;
        layers[l]->forward(h,bufs.fwd,B,T,s);
        cudaStreamSynchronize(s);
        e=cudaGetLastError();
        if(e!=cudaSuccess){fprintf(stderr,"layer %d: %s\n",l,cudaGetErrorString(e));return 1;}
    }
    bufs.fwd.save=base_save;

    auto*ln_f_w=w.get("transformer.ln_f.weight");
    if(ln_f_w)launch_pytorch_ln_kernel(h,(const float*)ln_f_w->data,n,d,1e-5f,s);
    const float*lm_w=nullptr;auto*lm_t=w.get("lm_head.weight");
    if(lm_t)lm_w=(const float*)lm_t->data;
    if(!lm_w)lm_w=wte;
    if(lm_w)launch_linear_fp32(h,lm_w,bufs.fwd.lm,n,V,d,s);
    cudaStreamSynchronize(s);
    e=cudaGetLastError();
    fprintf(stderr,"lm_head: %s\n",cudaGetErrorString(e));

    std::vector<float>cpu(n*V);
    cudaMemcpy(cpu.data(),bufs.fwd.lm,n*V*sizeof(float),cudaMemcpyDeviceToHost);
    float mx=0;for(auto v:cpu)if(fabs(v)>mx)mx=fabs(v);
    fprintf(stderr,"max_logit=%.6f\n",mx);

    w.free_all();bufs.free_all();
    cudaFree(ids);cudaFree(logits);cudaStreamDestroy(s);
    return mx>0?0:1;
}
