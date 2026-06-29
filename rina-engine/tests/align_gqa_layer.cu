#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "training/train.h"
#include "model.h"

static const int BLK2 = 256;
__global__ void add_k(float*c,const float*a,const float*b,int n){int i=blockIdx.x*BLK2+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
__global__ void silu_mul_k(float*o,const float*g,const float*u,int n){int i=blockIdx.x*BLK2+threadIdx.x;if(i<n)o[i]=(g[i]/(1.0f+expf(-g[i])))*u[i];}

static void dump(const char* name, const float* d, int n) {
    std::vector<float> cpu(n);
    cudaMemcpy(cpu.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost);
    char path[256]; snprintf(path,256,"/tmp/%s.bin",name);
    FILE*f=fopen(path,"wb"); fwrite(cpu.data(),sizeof(float),n,f); fclose(f);
}

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_rms_norm_fp32(float*,const float*,int,int,float,cudaStream_t);
extern void launch_rope_fp32(float*,const float*,const float*,int,int,int,int,cudaStream_t);
extern void build_qkv_fp32_kernel(const float*,const float*,const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,int,int,cudaStream_t);
extern void launch_flash_attn_fp32(const float*,const float*,const float*,float*,int,int,int,int,int,cudaStream_t);
extern void launch_transpose_attn(float*,const float*,int,int,int,cudaStream_t);

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/llama3.2-1b.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    if(layers.empty()){fprintf(stderr,"build fail\n");return 1;}

    int B=1,T=8,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int H=32,Hkv=8,dh=64,hd=8192;
    int ws=layers[0]->workspace_per_token(d,H,dh);
    int total=0; for(auto&l:layers)total+=l->saved_per_token(d,H,dh);
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    if(!bufs.fwd.h){fprintf(stderr,"alloc fail\n");return 1;}
    cudaStream_t s;cudaStreamCreate(&s);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={8270,1860,6390,6191,6734,7265,1466,5426};
    cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;

    float* h = bufs.fwd.h;
    float* a = bufs.fwd.a;
    float* m = bufs.fwd.m;
    int hq = n*H*dh, hk = n*Hkv*dh;

    // Embed
    launch_embedding_fp32(wte,d_ids,h,B,T,d,s); cudaStreamSynchronize(s);

    // LN1
    auto* ln1_w=(const float*)w.get("transformer.h.0.ln1.weight")->data;
    cudaMemcpyAsync(a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_rms_norm_fp32(a,ln1_w,n,d,1e-5f,s); cudaStreamSynchronize(s);
    dump("l1_ln1_out",a,n*d);

    // QKV
    auto*wq=(const float*)w.get("transformer.h.0.attn.w_q.weight")->data;
    auto*wk=(const float*)w.get("transformer.h.0.attn.w_k.weight")->data;
    auto*wv=(const float*)w.get("transformer.h.0.attn.w_v.weight")->data;
    launch_linear_fp32(a,wq,m,n,H*dh,d,s);
    launch_linear_fp32(a,wk,m+hq,n,Hkv*dh,d,s);
    launch_linear_fp32(a,wv,m+hq+hk,n,Hkv*dh,d,s); cudaStreamSynchronize(s);
    dump("l1_q",m,hq); dump("l1_k",m+hq,hk); dump("l1_v",m+hq+hk,hk);

    // RoPE QK
    auto*rqc=(const float*)w.get("transformer.h.0.attn.rope_q.cos")->data;
    auto*rqs=(const float*)w.get("transformer.h.0.attn.rope_q.sin")->data;
    auto*rkc=(const float*)w.get("transformer.h.0.attn.rope.cos")->data;
    auto*rks=(const float*)w.get("transformer.h.0.attn.rope.sin")->data;
    launch_rope_fp32(m,rqc,rqs,B,T,H,dh,s);
    launch_rope_fp32(m+hq,rkc,rks,B,T,Hkv,dh,s); cudaStreamSynchronize(s);
    dump("l1_q_rope",m,hq); dump("l1_k_rope",m+hq,hk);

    // Build QKV for FA
    float*Qf=m+hq+hk*2; float*Kf=Qf+n*H*dh; float*Vf=Kf+n*H*dh;
    build_qkv_fp32_kernel(m,m+hq,m+hq+hk,m,m+hq,Qf,Kf,Vf,B,T,H,Hkv,dh,0,dh,s);
    cudaStreamSynchronize(s);

    // FA
    launch_flash_attn_fp32(Qf,Kf,Vf,Qf,B,H,T,dh,dh,s); cudaStreamSynchronize(s);
    dump("l1_fa_out",Qf,n*H*dh);

    // Transpose + O proj
    launch_transpose_attn(a,Qf,H,T,dh,s);
    auto*wo=(const float*)w.get("transformer.h.0.attn.w_o.weight")->data;
    launch_linear_fp32(a,wo,a,n,d,H*dh,s); cudaStreamSynchronize(s);
    dump("l1_attn_out",a,n*d);

    // Residual add
    add_k<<<(n*d+BLK2-1)/BLK2,BLK2,0,s>>>(h,h,a,n*d); cudaStreamSynchronize(s);
    dump("l1_h_after_attn",h,n*d);

    // LN2
    auto*ln2_w=(const float*)w.get("transformer.h.0.ln2.weight")->data;
    cudaMemcpyAsync(a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_rms_norm_fp32(a,ln2_w,n,d,1e-5f,s); cudaStreamSynchronize(s);
    dump("l1_ln2_out",a,n*d);

    // MLP
    auto*w1=(const float*)w.get("transformer.h.0.mlp.w1.weight")->data;
    auto*w3=(const float*)w.get("transformer.h.0.mlp.w3.weight")->data;
    auto*w2=(const float*)w.get("transformer.h.0.mlp.w2.weight")->data;
    launch_linear_fp32(a,w1,m,n,hd,d,s);
    launch_linear_fp32(a,w3,m+n*hd,n,hd,d,s); cudaStreamSynchronize(s);
    dump("l1_gate",m,n*hd); dump("l1_up",m+n*hd,n*hd);
    silu_mul_k<<<(n*hd+BLK2-1)/BLK2,BLK2,0,s>>>(m,m,m+n*hd,n*hd);
    cudaStreamSynchronize(s); dump("l1_silu_mul",m,n*hd);
    launch_linear_fp32(m,w2,a,n,d,hd,s); cudaStreamSynchronize(s);
    dump("l1_mlp_out",a,n*d);

    add_k<<<(n*d+BLK2-1)/BLK2,BLK2,0,s>>>(h,h,a,n*d); cudaStreamSynchronize(s);
    dump("l1_h_final",h,n*d);
    fprintf(stderr,"Dumped to /tmp/l1_*.bin\n");
    w.free_all();bufs.free_all();cudaFree(d_ids);cudaStreamDestroy(s);
    return 0;
}
