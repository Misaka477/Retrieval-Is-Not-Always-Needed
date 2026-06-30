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
    char path[256]; snprintf(path,256,"/tmp/cmp_%s",name);
    FILE*f=fopen(path,"wb"); fwrite(cpu.data(),sizeof(float),n,f); fclose(f);
}

extern void launch_embedding_fp32(const float*,const int*,float*,int,int,int,cudaStream_t);
extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void launch_sigmoid_fp32(float*,int,cudaStream_t);
extern void launch_ssm_agg_fp32(const float*,const float*,const float*,
    const float*,const float*,const float*,float*,float*,int,int,int,cudaStream_t);
extern void launch_ssm_scan_fp32(const float*,const float*,float*,int,int,int,int,cudaStream_t);
extern void launch_silu_mul_inline(float*,const float*,const float*,int,cudaStream_t);

__global__ void add_f32_cmp(float*c,const float*a,const float*b,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
__global__ void silu_mul_cmp(float*o,const float*g,const float*u,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)o[i]=(g[i]/(1.0f+expf(-g[i])))*u[i];}

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/jamba_qw2.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    int B=1,T=3,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int H=cfg.n_heads,dh=cfg.head_dim,dc=cfg.d_c,ss=cfg.ssm_steps,hd=d*4*2/3/256*256;
    int ws=0,total=0;
    for(auto& l:layers){int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);}
    fprintf(stderr,"config: d=%d H=%d dh=%d dc=%d ss=%d hd=%d ws=%d\n",d,H,dh,dc,ss,hd,ws);

    // ─── Run via ENGINE forward ───
    BufferManager bufs1;
    bufs1.alloc_fwd(n,d,ws,hd,V,total);
    cudaStream_t s1;cudaStreamCreate(&s1);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={128000,9906,11}; cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;
    launch_embedding_fp32(wte,d_ids,bufs1.fwd.h,B,T,d,s1);
    cudaStreamSynchronize(s1);
    dump("emb",bufs1.fwd.h,n*d);
    float*bs=bufs1.fwd.save;
    for(int l=0;l<1;l++){bufs1.fwd.save=bs+layers[l]->save_offset*n;layers[l]->forward(bufs1.fwd.h,bufs1.fwd,B,T,s1);cudaStreamSynchronize(s1);}
    bufs1.fwd.save=bs;
    dump("eng_h",bufs1.fwd.h,n*d);
    fprintf(stderr,"engine forward done\n");

    // ─── Run via MANUAL step-by-step ───
    BufferManager bufs2;
    bufs2.alloc_fwd(n,d,ws,hd,V,total);
    cudaStream_t s2;cudaStreamCreate(&s2);
    launch_embedding_fp32(wte,d_ids,bufs2.fwd.h,B,T,d,s2);
    cudaStreamSynchronize(s2);

    const float*ln1_w=(const float*)w.get("transformer.h.0.ln1.weight")->data;
    const float*w_dq=(const float*)w.get("transformer.h.0.path.w_dq.weight")->data;
    const float*qn_w=(const float*)w.get("transformer.h.0.path.q_norm.weight")->data;
    const float*w_mem[3],*w_dec[3];
    for(int k=0;k<ss;k++){
        char bn[128];snprintf(bn,128,"transformer.h.0.path.w_mem.%d.weight",k);w_mem[k]=(const float*)w.get(bn)->data;
        snprintf(bn,128,"transformer.h.0.path.w_decay.%d.weight",k);w_dec[k]=(const float*)w.get(bn)->data;
    }
    const float*w_out_w=(const float*)w.get("transformer.h.0.path.w_out.weight")->data;
    const float*ln2_w=(const float*)w.get("transformer.h.0.ln2.weight")->data;
    const float*w1_w=(const float*)w.get("transformer.h.0.mlp.w1.weight")->data;
    const float*w2_w=(const float*)w.get("transformer.h.0.mlp.w2.weight")->data;
    const float*w3_w=(const float*)w.get("transformer.h.0.mlp.w3.weight")->data;

    int ncq=n*dc,ms=n*H*dh,doff=ncq+ss*ms,d2=d+H*dh;

    // LN1
    cudaMemcpyAsync(bufs2.fwd.a,bufs2.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s2);
    launch_pytorch_ln_kernel(bufs2.fwd.a,ln1_w,n,d,1e-5f,s2);
    cudaStreamSynchronize(s2);
    dump("man_a",bufs2.fwd.a,n*d);

    // cq
    launch_linear_fp32(bufs2.fwd.a,w_dq,bufs2.fwd.m,n,dc,d,s2);
    launch_pytorch_ln_kernel(bufs2.fwd.m,qn_w,n,dc,1e-5f,s2);
    cudaStreamSynchronize(s2);
    dump("man_cq",bufs2.fwd.m,n*dc);

    // SSM loop
    for(int k=0;k<ss;k++){
        launch_linear_fp32(bufs2.fwd.m,w_mem[k],bufs2.fwd.m+ncq+k*ms,n,H*dh,dc,s2);
        launch_linear_fp32(bufs2.fwd.m,w_dec[k],bufs2.fwd.m+doff+k*n*H,n,H,dc,s2);
        launch_sigmoid_fp32(bufs2.fwd.m+doff+k*n*H,n*H,s2);
    }
    cudaStreamSynchronize(s2);

    // agg
    float*da=bufs2.fwd.m+doff;
    float*ma=bufs2.fwd.m+doff+3*n*H;
    launch_ssm_agg_fp32(bufs2.fwd.m+ncq,bufs2.fwd.m+ncq+ms,bufs2.fwd.m+ncq+2*ms,
                        da,da+n*H,da+2*n*H,da,ma,H,dh,n,s2);
    cudaStreamSynchronize(s2);
    dump("man_sf",ma,n*H*dh);

    // scan
    launch_ssm_scan_fp32(ma,da,ma,B,T,H,dh,s2);
    cudaStreamSynchronize(s2);
    dump("man_sf_scanned",ma,n*H*dh);

    // concat (interleaved per-token)
    for(int i=0;i<n;i++){
        cudaMemcpyAsync(bufs2.fwd.m+i*d2,bufs2.fwd.a+i*d,d*sizeof(float),cudaMemcpyDeviceToDevice,s2);
        cudaMemcpyAsync(bufs2.fwd.m+i*d2+d,ma+i*H*dh,H*dh*sizeof(float),cudaMemcpyDeviceToDevice,s2);
    }
    cudaStreamSynchronize(s2);
    dump("man_concat",bufs2.fwd.m,n*d2);

    // w_out
    launch_linear_fp32(bufs2.fwd.m,w_out_w,bufs2.fwd.a,n,d,d2,s2);
    add_f32_cmp<<<(n*d+255)/256,256,0,s2>>>(bufs2.fwd.h,bufs2.fwd.h,bufs2.fwd.a,n*d);
    cudaStreamSynchronize(s2);
    dump("man_h_aft",bufs2.fwd.h,n*d);
    fprintf(stderr,"manual step-by-step done\n");

    // Compare
    std::vector<float> eng_h(n*d),man_h(n*d);
    cudaMemcpy(eng_h.data(),bufs1.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(man_h.data(),bufs2.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToHost);
    float max_diff=0;int max_i=0;
    for(int i=0;i<n*d;i++){float d=fabs(eng_h[i]-man_h[i]);if(d>max_diff){max_diff=d;max_i=i;}}
    fprintf(stderr,"Man vs Eng: max_diff=%.6e @ idx %d\n",max_diff,max_i);
    fprintf(stderr,"  eng[%d]=%f man[%d]=%f\n",max_i,eng_h[max_i],max_i,man_h[max_i]);

    w.free_all();bufs1.free_all();bufs2.free_all();cudaFree(d_ids);cudaStreamDestroy(s1);cudaStreamDestroy(s2);
    return 0;
}
