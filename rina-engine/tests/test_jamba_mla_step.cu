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
    char path[256]; snprintf(path,256,"/tmp/jm_%s",name);
    FILE*f=fopen(path,"wb"); fwrite(cpu.data(),sizeof(float),n,f); fclose(f);
}

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_rope_fp32(float*,const float*,const float*,int,int,int,int,cudaStream_t);
extern void build_qkv_fp32_kernel(const float*,const float*,const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,int,int,cudaStream_t);
extern void launch_flash_attn_fp32(const float*,const float*,const float*,float*,int,int,int,int,int,cudaStream_t);
extern void launch_transpose_attn(float*,const float*,int,int,int,cudaStream_t);
extern void launch_silu_mul_inline(float*,const float*,const float*,int,cudaStream_t);

__global__ void add_f32_jm(float*c,const float*a,const float*b,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/jamba_qw2.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    int l=3; // first MLA layer
    int B=1,T=3,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int H=cfg.n_heads, Hkv=cfg.n_kv_heads, dh=cfg.head_dim;
    int dhr=cfg.d_h_r?cfg.d_h_r:32, dc=cfg.d_c?cfg.d_c:160;
    int dq=dh+dhr, hd=d*4*2/3/256*256;
    fprintf(stderr,"d=%d H=%d Hkv=%d dh=%d dhr=%d dc=%d dq=%d hd=%d\n",d,H,Hkv,dh,dhr,dc,dq,hd);

    int ws=0,total=0;
    auto layers=build_layers(cfg,w);
    for(auto& l:layers){int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);}
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    cudaStream_t s;cudaStreamCreate(&s);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={128000,9906,11};
    cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;

    launch_embedding_fp32(wte,d_ids,bufs.fwd.h,B,T,d,s);
    cudaStreamSynchronize(s);

    // Run layers 0-2 (SSM) via layer->forward to get h for layer 3
    float*base_save=bufs.fwd.save;
    for(int i=0;i<l;i++){
        bufs.fwd.save=base_save+layers[i]->save_offset*n;
        layers[i]->forward(bufs.fwd.h,bufs.fwd,B,T,s);
        cudaStreamSynchronize(s);
    }
    fprintf(stderr,"layers 0-2 done\n");

    // Weights for layer 3 (MLA)
    const float*ln1_w=(const float*)w.get("transformer.h.3.ln1.weight")->data;
    const float*ln2_w=(const float*)w.get("transformer.h.3.ln2.weight")->data;
    const float*w_dqkv=(const float*)w.get("transformer.h.3.path.w_dqkv.weight")->data;
    const float*qn_w=(const float*)w.get("transformer.h.3.path.q_norm.weight")->data;
    const float*w_uq=(const float*)w.get("transformer.h.3.path.w_uq.weight")->data;
    const float*w_uk=(const float*)w.get("transformer.h.3.path.w_uk.weight")->data;
    const float*w_k2v=(const float*)w.get("transformer.h.3.path.w_k2v.weight")->data;
    const float*w_qr=(const float*)w.get("transformer.h.3.path.w_qr.weight")->data;
    const float*w_kr=(const float*)w.get("transformer.h.3.path.w_kr.weight")->data;
    const float*c_proj=(const float*)w.get("transformer.h.3.path.c_proj.weight")->data;
    const float*rqc=(const float*)w.get("transformer.h.3.path.rope_q.cos")->data;
    const float*rqs=(const float*)w.get("transformer.h.3.path.rope_q.sin")->data;
    const float*rc=(const float*)w.get("transformer.h.3.path.rope.cos")->data;
    const float*rs=(const float*)w.get("transformer.h.3.path.rope.sin")->data;
    const float*w1_w=(const float*)w.get("transformer.h.3.mlp.w1.weight")->data;
    const float*w2_w=(const float*)w.get("transformer.h.3.mlp.w2.weight")->data;
    const float*w3_w=(const float*)w.get("transformer.h.3.mlp.w3.weight")->data;

    int oq=n*H*dh, ov=oq+n*Hkv*dh, oqr=ov+n*Hkv*dh;

    // LN1
    cudaMemcpyAsync(bufs.fwd.a,bufs.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_pytorch_ln_kernel(bufs.fwd.a,ln1_w,n,d,1e-5f,s);
    cudaStreamSynchronize(s);
    dump("l3_h_in",bufs.fwd.h,n*d);
    dump("l3_a_ln1",bufs.fwd.a,n*d);
    fprintf(stderr,"ln1\n");

    // Save LN1 for qr/kr
    cudaMemcpyAsync(bufs.fwd.save,bufs.fwd.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    cudaStreamSynchronize(s);

    // cq
    launch_linear_fp32(bufs.fwd.a,w_dqkv,bufs.fwd.m,n,dc,d,s);
    launch_pytorch_ln_kernel(bufs.fwd.m,qn_w,n,dc,1e-5f,s);
    cudaStreamSynchronize(s);
    dump("l3_cq",bufs.fwd.m,n*dc);

    // qc, kc, v
    launch_linear_fp32(bufs.fwd.m,w_uq,bufs.fwd.a,n,H*dh,dc,s);
    launch_linear_fp32(bufs.fwd.m,w_uk,bufs.fwd.m+oq,n,Hkv*dh,dc,s);
    launch_linear_fp32(bufs.fwd.m+oq,w_k2v,bufs.fwd.m+ov,n,Hkv*dh,Hkv*dh,s);
    cudaStreamSynchronize(s);
    dump("l3_qc",bufs.fwd.a,n*H*dh);
    dump("l3_kc",bufs.fwd.m+oq,n*Hkv*dh);
    dump("l3_v",bufs.fwd.m+ov,n*Hkv*dh);
    fprintf(stderr,"qkv\n");

    // qr, kr from saved LN1
    launch_linear_fp32(bufs.fwd.save,w_qr,bufs.fwd.m+oqr,n,H*dhr,d,s);
    launch_linear_fp32(bufs.fwd.save,w_kr,bufs.fwd.m+oqr+n*H*dhr,n,Hkv*dhr,d,s);
    cudaStreamSynchronize(s);
    dump("l3_qr",bufs.fwd.m+oqr,n*H*dhr);
    dump("l3_kr",bufs.fwd.m+oqr+n*H*dhr,n*Hkv*dhr);

    // RoPE
    if(rqc)launch_rope_fp32(bufs.fwd.m+oqr,rqc,rqs,B,T,H,dhr,s);
    if(rc)launch_rope_fp32(bufs.fwd.m+oqr+n*H*dhr,rc,rs,B,T,Hkv,dhr,s);
    cudaStreamSynchronize(s);
    dump("l3_qr_rope",bufs.fwd.m+oqr,n*H*dhr);
    dump("l3_kr_rope",bufs.fwd.m+oqr+n*H*dhr,n*Hkv*dhr);
    fprintf(stderr,"rope\n");

    // Build QKV
    float* Qf=bufs.fwd.m+oqr+n*H*dhr+n*Hkv*dhr;
    float* Kf=Qf+n*H*dq;
    float* Vf=Kf+n*H*dq;
    build_qkv_fp32_kernel(bufs.fwd.a,bufs.fwd.m+oq,bufs.fwd.m+ov,bufs.fwd.m+oqr,bufs.fwd.m+oqr+n*H*dhr,
                          Qf,Kf,Vf,B,T,H,Hkv,dh,dhr,dq,s);
    cudaStreamSynchronize(s);
    dump("l3_Qf",Qf,n*H*dq); dump("l3_Kf",Kf,n*H*dq); dump("l3_Vf",Vf,n*H*dh);

    // Flash attention
    launch_flash_attn_fp32(Qf,Kf,Vf,Qf,B,H,T,dq,dh,s);
    cudaStreamSynchronize(s);

    // Transpose + c_proj
    launch_transpose_attn(bufs.fwd.a,Qf,H,T,dh,s);
    launch_linear_fp32(bufs.fwd.a,c_proj,bufs.fwd.a,n,d,H*dh,s);
    add_f32_jm<<<(n*d+255)/256,256,0,s>>>(bufs.fwd.h,bufs.fwd.h,bufs.fwd.a,n*d);
    cudaStreamSynchronize(s);
    dump("l3_h_aft",bufs.fwd.h,n*d);
    fprintf(stderr,"attn\n");

    // LN2 + MLP
    cudaMemcpyAsync(bufs.fwd.a,bufs.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_pytorch_ln_kernel(bufs.fwd.a,ln2_w,n,d,1e-5f,s);
    launch_linear_fp32(bufs.fwd.a,w1_w,bufs.fwd.m,n,hd,d,s);
    launch_linear_fp32(bufs.fwd.a,w3_w,bufs.fwd.m+n*hd,n,hd,d,s);
    launch_silu_mul_inline(bufs.fwd.m,bufs.fwd.m,bufs.fwd.m+n*hd,n*hd,s);
    launch_linear_fp32(bufs.fwd.m,w2_w,bufs.fwd.a,n,d,hd,s);
    add_f32_jm<<<(n*d+255)/256,256,0,s>>>(bufs.fwd.h,bufs.fwd.h,bufs.fwd.a,n*d);
    cudaStreamSynchronize(s);
    dump("l3_h_final",bufs.fwd.h,n*d);
    fprintf(stderr,"mlp\n");

    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){fprintf(stderr,"err: %s\n",cudaGetErrorString(e));return 1;}

    w.free_all();bufs.free_all();cudaFree(d_ids);cudaStreamDestroy(s);
    return 0;
}
