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
    char path[256]; snprintf(path,256,"/tmp/a_%s",name);
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

__global__ void add_f32_ma(float*c,const float*a,const float*b,int n){int i=blockIdx.x*256+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}

int main() {
    ModelConfig cfg; TensorMap w;
    if(!load_model("/tmp/model_a.rinn",cfg,w)){fprintf(stderr,"load fail\n");return 1;}
    auto layers=build_layers(cfg,w);
    if(layers.empty())return 1;

    int B=1,T=3,n=B*T,d=cfg.dim,V=cfg.vocab_size;
    int H=cfg.n_heads, Hkv=cfg.n_kv_heads, dh=cfg.head_dim;
    int dhr=cfg.d_h_r?cfg.d_h_r:32, dc=cfg.d_c?cfg.d_c:160;
    int dq=dh+dhr, hd=d*4*2/3/256*256;
    fprintf(stderr,"d=%d H=%d Hkv=%d dh=%d dhr=%d dc=%d dq=%d hd=%d\n",d,H,Hkv,dh,dhr,dc,dq,hd);

    int ws=0,total=0;
    for(auto& l:layers){
        int w=l->workspace_per_token(d,cfg.n_heads,cfg.head_dim);if(w>ws)ws=w;
        total+=l->saved_per_token(d,cfg.n_heads,cfg.head_dim);
    }
    BufferManager bufs;
    bufs.alloc_fwd(n,d,ws,hd,V,total);
    cudaStream_t s;cudaStreamCreate(&s);
    int*d_ids;cudaMalloc(&d_ids,n*sizeof(int));
    int ids[]={128000,9906,11};
    cudaMemcpy(d_ids,ids,n*sizeof(int),cudaMemcpyHostToDevice);
    const float*wte=(const float*)w.get("transformer.wte.weight")->data;

    launch_embedding_fp32(wte,d_ids,bufs.fwd.h,B,T,d,s);
    cudaStreamSynchronize(s);
    dump("embed",bufs.fwd.h,n*d);
    fprintf(stderr,"embed\n");

    // Weights for layer 0
    const float*ln1_w=(const float*)w.get("transformer.h.0.ln1.weight")->data;
    const float*ln2_w=(const float*)w.get("transformer.h.0.ln2.weight")->data;
    const float*w_dqkv=(const float*)w.get("transformer.h.0.path.w_dqkv.weight")->data;
    const float*qn_w=(const float*)w.get("transformer.h.0.path.q_norm.weight")->data;
    const float*kn_w=(const float*)w.get("transformer.h.0.path.k_norm.weight")->data;
    const float*w_uq=(const float*)w.get("transformer.h.0.path.w_uq.weight")->data;
    const float*w_uk=(const float*)w.get("transformer.h.0.path.w_uk.weight")->data;
    const float*w_k2v=(const float*)w.get("transformer.h.0.path.w_k2v.weight")->data;
    const float*w_qr=(const float*)w.get("transformer.h.0.path.w_qr.weight")->data;
    const float*w_kr=(const float*)w.get("transformer.h.0.path.w_kr.weight")->data;
    const float*c_proj=(const float*)w.get("transformer.h.0.path.c_proj.weight")->data;
    const float*rqc=(const float*)w.get("transformer.h.0.path.rope_q.cos")->data;
    const float*rqs=(const float*)w.get("transformer.h.0.path.rope_q.sin")->data;
    const float*rc=(const float*)w.get("transformer.h.0.path.rope.cos")->data;
    const float*rs=(const float*)w.get("transformer.h.0.path.rope.sin")->data;
    const float*w1_w=(const float*)w.get("transformer.h.0.mlp.w1.weight")->data;
    const float*w2_w=(const float*)w.get("transformer.h.0.mlp.w2.weight")->data;
    const float*w3_w=(const float*)w.get("transformer.h.0.mlp.w3.weight")->data;

    int oq=n*H*dh, ov=oq+n*Hkv*dh, oqr=ov+n*Hkv*dh;

    // Step 1: LN1 → a
    cudaMemcpyAsync(bufs.fwd.a,bufs.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_pytorch_ln_kernel(bufs.fwd.a,ln1_w,n,d,1e-5f,s);
    cudaStreamSynchronize(s);
    dump("l0_a_ln1",bufs.fwd.a,n*d);
    // SAVE LN1 output BEFORE it gets overwritten by w_uq
    cudaMemcpyAsync(bufs.fwd.save,bufs.fwd.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    cudaStreamSynchronize(s);
    fprintf(stderr,"ln1\n");

    // Step 2: w_dqkv → q_norm → cq
    launch_linear_fp32(bufs.fwd.a,w_dqkv,bufs.fwd.m,n,dc,d,s);
    launch_pytorch_ln_kernel(bufs.fwd.m,qn_w,n,dc,1e-5f,s);
    cudaStreamSynchronize(s);
    dump("l0_cq",bufs.fwd.m,n*dc);
    fprintf(stderr,"cq\n");

    // Step 3: w_uq → qc, w_uk → kc, w_k2v → v
    launch_linear_fp32(bufs.fwd.m,w_uq,bufs.fwd.a,n,H*dh,dc,s);
    cudaStreamSynchronize(s);
    dump("l0_qc",bufs.fwd.a,n*H*dh);
    launch_linear_fp32(bufs.fwd.m,w_uk,bufs.fwd.m+oq,n,Hkv*dh,dc,s);
    cudaStreamSynchronize(s);
    dump("l0_kc",bufs.fwd.m+oq,n*Hkv*dh);
    launch_linear_fp32(bufs.fwd.m+oq,w_k2v,bufs.fwd.m+ov,n,Hkv*dh,Hkv*dh,s);
    cudaStreamSynchronize(s);
    dump("l0_v",bufs.fwd.m+ov,n*Hkv*dh);
    fprintf(stderr,"qkv\n");

    // Step 4: RoPE (qr, kr) from LN1 output saved in bufs.fwd.a
    // Need to re-save LN1 output since bufs.fwd.a is now qc
    // Actually, the engine stores LN1 output in save buffer (off_lp = d)
    // And reads from save+off_lp*n for w_qr/w_kr
    cudaMemcpyAsync(bufs.fwd.save+d*n,bufs.fwd.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s); // save ln1 copy
    // For qr: w_qr is applied to the LN1 copy (save+off_lp*n)
    // Actually the engine does:
    //   launch_linear_fp32(b.save+off_lp*n, w_qr, b.m+oqr, n, H*dhr, d, s);
    // So input from save+off_lp*n (saved LN1), output to bufs.fwd.m+oqr
    // But we stored cq at bufs.fwd.m[0]... need a different approach
    // Let me re-read the engine code more carefully...

    // Actually looking at the engine forward:
    // 1. Copy h to save+off_l1 (h input)
    // 2. Copy h to b.a, LN1 on b.a → b.a has ln1 output
    // 3. Save b.a to save+off_lp (ln1 output copy)
    // 4. w_dqkv(b.a) → b.m, q_norm on b.m → b.m has cq
    // 5. Save cq to save+off_cq
    // 6. w_uq(b.m) → b.a (b.a now has qc)
    // 7. w_uk(b.m) → b.m+oq (kc)
    // 8. w_k2v(b.m+oq) → b.m+ov (v)
    // 9. w_qr(save+off_lp) → b.m+oqr (qr) — reads from saved LN1
    // 10. w_kr(save+off_lp) → b.m+okr (kr) — reads from saved LN1
    // 11. rope on qr, kr
    // 12. build_qkv → Qf, Kf, Vf
    // 13. flash_attn
    // 14. transpose + c_proj

    // Step 5: w_qr(ln1_out) → qr, w_kr(ln1_out) → kr
    // (LN1 output was saved to bufs.fwd.save in step 1)
    launch_linear_fp32(bufs.fwd.save,w_qr,bufs.fwd.m+oqr,n,H*dhr,d,s);
    launch_linear_fp32(bufs.fwd.save,w_kr,bufs.fwd.m+oqr+n*H*dhr,n,Hkv*dhr,d,s);
    cudaStreamSynchronize(s);
    dump("l0_qr",bufs.fwd.m+oqr,n*H*dhr);
    dump("l0_kr",bufs.fwd.m+oqr+n*H*dhr,n*Hkv*dhr);
    fprintf(stderr,"qr,kr\n");

    // Step 6: RoPE
    if(rqc)launch_rope_fp32(bufs.fwd.m+oqr,rqc,rqs,B,T,H,dhr,s);
    if(rc)launch_rope_fp32(bufs.fwd.m+oqr+n*H*dhr,rc,rs,B,T,Hkv,dhr,s);
    cudaStreamSynchronize(s);
    dump("l0_qr_rope",bufs.fwd.m+oqr,n*H*dhr);
    dump("l0_kr_rope",bufs.fwd.m+oqr+n*H*dhr,n*Hkv*dhr);
    fprintf(stderr,"rope\n");

    // Step 7: build QKV
    // buf layout: ab = qf(2xH*dq) at b.m+okr + ...
    // Let's use the workspace at b.m properly
    float* Qf=bufs.fwd.m+oqr+n*H*dhr+n*Hkv*dhr;
    float* Kf=Qf+n*H*dq;
    float* Vf=Kf+n*H*dq;
    // Need to get qc back (stored at bufs.fwd.a earlier, but now bufs.fwd.a has ln1+something)
    // Actually, qc was written to bufs.fwd.a in step 3, but then overwritten.
    // Let me read from the dump we made
    // Actually bufs.fwd.a was used for qc, but we need to rederive it
    // The engine's build_qkv reads from b.a (qc), b.m+oq (kc), b.m+ov (v), 
    //   b.m+oqr (qr), b.m+okr (kr)
    // Problem: b.a has the qc but we need it. Let me check what's in b.a now...
    // In the engine forward: after w_uq writes to b.a, the next use is build_qkv.
    // Between w_uq and build_qkv, only these are called:
    //   launch_linear_fp32(b.m,w_uk,b.m+oq,...)
    //   launch_linear_fp32(b.m+oq,w_k2v,b.m+ov,...)
    //   launch_linear_fp32(b.save+off_lp*n,w_qr,b.m+oqr,...)
    //   launch_linear_fp32(b.save+off_lp*n,w_kr,b.m+okr,...)
    //   launch_rope_fp32(b.m+oqr,...)
    //   launch_rope_fp32(b.m+okr,...)
    // Dump qc right before build_qkv (verify it hasn't been modified)
    cudaStreamSynchronize(s);
    dump("l0_qc_final",bufs.fwd.a,n*H*dh);
    // Build QKV (engine's build_qkv_fp32_kernel)
    build_qkv_fp32_kernel(bufs.fwd.a,bufs.fwd.m+oq,bufs.fwd.m+ov,bufs.fwd.m+oqr,bufs.fwd.m+oqr+n*H*dhr,
                          Qf,Kf,Vf,B,T,H,Hkv,dh,dhr,dq,s);
    cudaStreamSynchronize(s);
    cudaStreamSynchronize(s);
    dump("l0_Qf",Qf,n*H*dq);
    dump("l0_Kf",Kf,n*H*dq);
    dump("l0_Vf",Vf,n*H*dh);
    fprintf(stderr,"build_qkv\n");

    // Step 8: Flash attention
    launch_flash_attn_fp32(Qf,Kf,Vf,Qf,B,H,T,dq,dh,s);
    cudaStreamSynchronize(s);
    dump("l0_attn_out",Qf,n*H*dh); // attn output in Qf
    fprintf(stderr,"flash_attn\n");

    // Step 9: Transpose + c_proj
    launch_transpose_attn(bufs.fwd.a,Qf,H,T,dh,s);  // write to b.a
    cudaStreamSynchronize(s);
    dump("l0_attn_transposed",bufs.fwd.a,n*H*dh);
    launch_linear_fp32(bufs.fwd.a,c_proj,bufs.fwd.a,n,d,H*dh,s);
    cudaStreamSynchronize(s);
    dump("l0_cproj_out",bufs.fwd.a,n*d);
    // Add residual: h += bufs.fwd.a
    add_f32_ma<<<(n*d+255)/256,256,0,s>>>(bufs.fwd.h,bufs.fwd.h,bufs.fwd.a,n*d);
    cudaStreamSynchronize(s);
    dump("l0_h_aft_attn",bufs.fwd.h,n*d);
    fprintf(stderr,"c_proj\n");

    // Step 10: LN2 + MLP
    cudaMemcpyAsync(bufs.fwd.a,bufs.fwd.h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
    launch_pytorch_ln_kernel(bufs.fwd.a,ln2_w,n,d,1e-5f,s);
    cudaStreamSynchronize(s);
    dump("l0_ln2_out",bufs.fwd.a,n*d);
    launch_linear_fp32(bufs.fwd.a,w1_w,bufs.fwd.m,n,hd,d,s);
    launch_linear_fp32(bufs.fwd.a,w3_w,bufs.fwd.m+n*hd,n,hd,d,s);
    cudaStreamSynchronize(s);
    dump("l0_gu",bufs.fwd.m,2*n*hd);
    launch_silu_mul_inline(bufs.fwd.m,bufs.fwd.m,bufs.fwd.m+n*hd,n*hd,s);
    launch_linear_fp32(bufs.fwd.m,w2_w,bufs.fwd.a,n,d,hd,s);
    add_f32_ma<<<(n*d+255)/256,256,0,s>>>(bufs.fwd.h,bufs.fwd.h,bufs.fwd.a,n*d);
    cudaStreamSynchronize(s);
    dump("l0_h_final",bufs.fwd.h,n*d);
    fprintf(stderr,"mlp\n");

    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){fprintf(stderr,"err: %s\n",cudaGetErrorString(e));return 1;}

    w.free_all();bufs.free_all();cudaFree(d_ids);cudaStreamDestroy(s);
    return 0;
}
