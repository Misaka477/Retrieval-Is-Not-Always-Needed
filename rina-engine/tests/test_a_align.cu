#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "core/layer.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

static void dump(const char* name, const float* d, int n) {
    std::vector<float> cpu(n);
    cudaMemcpy(cpu.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost);
    char path[256]; snprintf(path,256,"/tmp/%s",name);
    FILE*f=fopen(path,"wb"); fwrite(cpu.data(),sizeof(float),n,f); fclose(f);
}

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/model_a.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    if(layers.empty()){fprintf(stderr,"build fail\n");return 1;}
    fprintf(stderr,"Model: %s dim=%d layers=%zu\n",cfg.name.c_str(),cfg.dim,layers.size());

    int B=1,T=3,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int ws=0,total=0;
    for(auto& l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    int hd=d*4*2/3/256*256;
    fprintf(stderr,"ws=%d total_saved=%d hd=%d\n",ws,total,hd);

    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    if(!bufs.fwd.h){fprintf(stderr,"alloc fail\n");return 1;}
    cudaStream_t s;cudaStreamCreate(&s);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={128000,9906,11};
    cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;

    launch_embedding_fp32(wte,d_ids,bufs.fwd.h,B,T,d,s);
    cudaStreamSynchronize(s);
    fprintf(stderr,"embed: %s\n",cudaGetErrorString(cudaGetLastError()));

    float*base_save=bufs.fwd.save;
    for(int l=0;l<(int)layers.size();l++){
        bufs.fwd.save=base_save+layers[l]->save_offset*n;
        layers[l]->forward(bufs.fwd.h,bufs.fwd,B,T,s);
        cudaStreamSynchronize(s);
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess){fprintf(stderr,"layer %d: %s\n",l,cudaGetErrorString(e));return 1;}
    }
    bufs.fwd.save=base_save;

    // Final norm (LayerNorm for L3X, not RMSNorm)
    auto*ln_f=w.get("transformer.ln_f.weight");
    if(ln_f){
        launch_pytorch_ln_kernel(bufs.fwd.h,(const float*)ln_f->data,n,d,1e-5f,s);
    }
    const float*lm_w=(const float*)w.get("lm_head.weight")->data;
    launch_linear_fp32(bufs.fwd.h,lm_w,bufs.fwd.lm,n,V,d,s);
    cudaStreamSynchronize(s);
    if(cudaGetLastError()!=cudaSuccess){fprintf(stderr,"lm_head fail\n");return 1;}

    dump("a_logits.bin",bufs.fwd.lm,n*V);
    fprintf(stderr,"Dumped logits\n");

    w.free_all();bufs.free_all();cudaFree(d_ids);cudaStreamDestroy(s);
    return 0;
}
