#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

#define CHECK(cmd) do { cudaError_t e = cmd; if(e!=cudaSuccess){fprintf(stderr,"  ERR at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}} while(0)

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/llama3.2-1b.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    if(layers.empty()){fprintf(stderr,"build fail\n");return 1;}

    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size,hd=8192;
    int ws=0,total=0;
    for(auto&l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim); if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    if(!bufs.fwd.h){fprintf(stderr,"alloc fail\n");return 1;}

    cudaStream_t s;cudaStreamCreate(&s);
    int*ids;float*logits;
    cudaMalloc(&ids,n*sizeof(int)); cudaMalloc(&logits,n*V*sizeof(float));
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    float*base_save=bufs.fwd.save;
    launch_embedding_fp32(wte,ids,bufs.fwd.h,B,T,d,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"embed OK\n");

    // Single layer forward with per-step error checking
    int l=0;
    auto&layer=layers[l];
    bufs.fwd.save=base_save+layer->save_offset*n;
    float*h=bufs.fwd.h;
    auto&b=bufs.fwd;
    
    // Step by step
    int ns=n*d;
    cudaMemcpyAsync(b.save+0,h,ns*sizeof(float),cudaMemcpyDeviceToDevice,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"save h OK\n");
    
    cudaMemcpyAsync(b.a,h,ns*sizeof(float),cudaMemcpyDeviceToDevice,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"copy h->a OK\n");
    
    // Check if it's RMSNorm vs LayerNorm
    auto* ln1 = w.get("transformer.h.0.ln1.weight");
    fprintf(stderr,"ln1.weight: ptr=%p n_elems=%d\n",ln1?ln1->data:0,ln1?ln1->n_elems:0);
    
    extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
    launch_rms_norm_fp32(b.a,(const float*)ln1->data,n,d,1e-5f,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"rms_norm OK\n");
    
    // Q projection
    auto* wq = w.get("transformer.h.0.attn.w_q.weight");
    fprintf(stderr,"w_q: ptr=%p shape=[%d,%d]\n",wq->data,wq->shape[0],wq->shape[1]);
    launch_linear_fp32(b.a,(const float*)wq->data,b.m,n,32*64,d,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"Q proj OK\n");
    
    // K projection
    auto* wk = w.get("transformer.h.0.attn.w_k.weight");
    launch_linear_fp32(b.a,(const float*)wk->data,b.m+8*64,n,8*64,d,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"K proj OK\n");
    
    // V projection
    auto* wv = w.get("transformer.h.0.attn.w_v.weight");
    launch_linear_fp32(b.a,(const float*)wv->data,b.m+8*64+8*64,n,8*64,d,s); CHECK(cudaStreamSynchronize(s));
    fprintf(stderr,"V proj OK\n");
    
    // RoPE
    auto* rqc = w.get("transformer.h.0.attn.rope_q.cos");
    auto* rqs = w.get("transformer.h.0.attn.rope_q.sin");
    fprintf(stderr,"rope_q: cos=%p sin=%p\n",rqc?rqc->data:0,rqs?rqs->data:0);
    if(rqc&&rqs){
        extern void launch_rope_fp32(float*,const float*,const float*,int,int,int,int,cudaStream_t);
        launch_rope_fp32(b.m,(const float*)rqc->data,(const float*)rqs->data,B,T,32,64,s);
        CHECK(cudaStreamSynchronize(s));
        fprintf(stderr,"RoPE Q OK\n");
        auto* rkc=w.get("transformer.h.0.attn.rope.cos");
        auto* rks=w.get("transformer.h.0.attn.rope.sin");
        launch_rope_fp32(b.m+8*64,(const float*)rkc->data,(const float*)rks->data,B,T,8,64,s);
        CHECK(cudaStreamSynchronize(s));
        fprintf(stderr,"RoPE K OK\n");
    }

    fprintf(stderr,"ALL STEP BY STEP OK\n");
    w.free_all();bufs.free_all();
    cudaFree(ids);cudaFree(logits);cudaStreamDestroy(s);
    return 0;
}
