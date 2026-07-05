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
extern void launch_linear_dispatch(const void*,QuantType,const float*,float*,int,int,int,cudaStream_t);
int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/llama3.2-1b.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    fprintf(stderr,"Model: dim=%d layers=%d heads=%d kv=%d V=%d\n",
        cfg.dim,cfg.n_layers,cfg.n_heads,cfg.n_kv_heads,cfg.vocab_size);
    auto layers=build_layers(cfg,w);
    if(layers.empty()){fprintf(stderr,"build layers fail\n");return 1;}
    fprintf(stderr,"%zu layers built\n",layers.size());

    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int hd=cfg.dim*4*2/3/256*256;
    int ws=0,total=0;
    for(auto&l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim); if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    fprintf(stderr,"ws=%d saved=%d\n",ws,total);
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    if(!bufs.fwd.h){fprintf(stderr,"alloc fail\n");return 1;}

    cudaStream_t s;cudaStreamCreate(&s);
    int*ids;float*logits;
    cudaMalloc(&ids,n*sizeof(int));
    cudaMalloc(&logits,n*V*sizeof(float));
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    float*base_save=bufs.fwd.save;
    launch_embedding_fp32(wte,ids,bufs.fwd.h,B,T,d,s);
    cudaStreamSynchronize(s);
    fprintf(stderr,"embed: %s\n",cudaGetErrorString(cudaGetLastError()));

    for(int l=0;l<(int)layers.size();l++){
        cudaError_t prior=cudaGetLastError(); // clear stale errors
        bufs.fwd.save=base_save+layers[l]->save_offset*n;
        fprintf(stderr,"layer %d forward... ",l); fflush(stderr);
        layers[l]->forward(bufs.fwd.h,bufs.fwd,B,T,s);
        cudaStreamSynchronize(s);
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess){
            fprintf(stderr,"FAIL: %s (prior=%s)\n",cudaGetErrorString(e),cudaGetErrorString(prior));
            return 1;
        }
        fprintf(stderr,"OK\n");
    }
    bufs.fwd.save=base_save;

    auto*ln_f=w.get("transformer.ln_f.weight");
    if(ln_f)launch_pytorch_ln_kernel(bufs.fwd.h,(const float*)ln_f->data,n,d,1e-5f,s);
    auto*lm_t=w.get("lm_head.weight");
    if(lm_t) {
        launch_linear_dispatch(lm_t->data,lm_t->quant_type,bufs.fwd.h,bufs.fwd.lm,n,V,d,s);
    } else {
        launch_linear_fp32(bufs.fwd.h,wte,bufs.fwd.lm,n,V,d,s);
    }
    cudaStreamSynchronize(s);
    cudaError_t e=cudaGetLastError();
    fprintf(stderr,"lm_head: %s\n",cudaGetErrorString(e));

    std::vector<float>cpu(n*V);
    cudaMemcpy(cpu.data(),bufs.fwd.lm,n*V*sizeof(float),cudaMemcpyDeviceToHost);
    float mx=0; for(auto v:cpu)if(fabs(v)>mx)mx=fabs(v);
    fprintf(stderr,"max_logit=%.6f\n",mx);
    w.free_all();bufs.free_all();
    cudaFree(ids);cudaFree(logits);cudaStreamDestroy(s);
    return mx>0?0:1;
}
