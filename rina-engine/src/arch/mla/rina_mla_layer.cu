#include "core/layer.h"
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "kernels/gemm.cuh"
#include <cstdio>
#include <cmath>
#include <string>

extern void launch_linear_fp32(const float*,const float*,float*,int,int,int,cudaStream_t);
extern void launch_linear_dispatch(const void*,QuantType,const float*,float*,int,int,int,cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*,const float*,int,int,float,cudaStream_t);
extern void build_qkv_fp32_kernel(const float*,const float*,const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,int,int,cudaStream_t);
extern void build_qkv_bwd_kernel(const float*,const float*,const float*,float*,float*,float*,float*,float*,int,int,int,int,int,int,int,cudaStream_t);
extern void launch_flashattn_fwd_save_stats(const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,cudaStream_t);
extern void launch_flash_attn_bwd_fp32(const float*,const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,float*,int,int,int,int,int,cudaStream_t);
extern void launch_transpose_attn(float*,const float*,int,int,int,cudaStream_t);
extern void launch_rope_fp32_hf(float*,const float*,const float*,int,int,int,int,cudaStream_t);
extern void launch_rope_bwd_fp32_hf(float*,const float*,const float*,const float*,int,int,int,int,cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*,const float*,const float*,float*,float*,int,int,cudaStream_t);
extern void launch_silu_mul_bwd_fp32(const float*,const float*,const float*,float*,float*,int,cudaStream_t);
extern void launch_silu_mul_inline(float*,const float*,const float*,int,cudaStream_t);
extern void launch_copy_f32(float*,const float*,int,cudaStream_t);

static const int BLK=256;
__global__ void add_f32_m(float*c,const float*a,const float*b,int n){int i=blockIdx.x*BLK+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
__global__ void silu_mul_f32_m(float*o,const float*g,const float*u,int n){int i=blockIdx.x*BLK+threadIdx.x;if(i<n)o[i]=(g[i]/(1.0f+expf(-g[i])))*u[i];}

static std::string _tn(int l,const char*c,const char*p){char b[128];snprintf(b,128,"transformer.h.%d.%s.%s",l,c,p);return b;}
static size_t wg_of(const TensorMap& w,const std::string& n_){size_t o=0;for(auto&[n,wt]:w.tensors){if(wt.quant_type!=QuantType::FP32)continue;if(n==n_)return o;o+=wt.n_elems;}return (size_t)-1;}

struct RinaMLAImpl {
    int d,H,Hkv,dh,dhr,dq,dc,hd;
    int kv_layer_idx = 0;
    const float *q_norm_w,*k_norm_w,*ln1_w,*ln2_w;
    const float *rqc,*rqs,*rc,*rs;
    WeightRef w_dqkv,w_uq,w_uk,w_k2v,w_qr,w_kr,c_proj;
    WeightRef w1,w2,w3;
    size_t off_dqkv,off_qn,off_uq,off_uk,off_k2v,off_qr,off_kr,off_proj,off_1,off_2,off_3,off_ln1,off_ln2;
    int off_l1,off_cq,off_lp,off_l2i,off_l2o,off_gu;

    bool init(const ModelConfig& cfg,const TensorMap& w,int l) {
        d=cfg.dim;H=cfg.n_heads;Hkv=cfg.n_kv_heads;dh=cfg.head_dim;dhr=cfg.d_h_r?cfg.d_h_r:32;dq=dh+dhr;dc=cfg.d_c?cfg.d_c:160;hd=d*4*2/3/256*256;
        kv_layer_idx = l;
        auto ld=[&](const std::string& n,const float*&p){auto*t=w.get(n);if(!t)return false;p=(const float*)t->data;return true;};
        auto ld_ref=[&](const std::string& n,WeightRef& ref){auto*t=w.get(n);if(!t)return false;ref.data=t->data;ref.qt=t->quant_type;return true;};
        ld_ref(_tn(l,"path","w_dqkv.weight"),w_dqkv);        ld(_tn(l,"path","q_norm.weight"),q_norm_w);
        ld(_tn(l,"path","k_norm.weight"),k_norm_w);
        ld_ref(_tn(l,"path","w_uq.weight"),w_uq);ld_ref(_tn(l,"path","w_uk.weight"),w_uk);ld_ref(_tn(l,"path","w_k2v.weight"),w_k2v);
        ld_ref(_tn(l,"path","w_qr.weight"),w_qr);ld_ref(_tn(l,"path","w_kr.weight"),w_kr);ld_ref(_tn(l,"path","c_proj.weight"),c_proj);
        ld(_tn(l,"path","rope_q.cos"),rqc);ld(_tn(l,"path","rope_q.sin"),rqs);
        ld(_tn(l,"path","rope.cos"),rc);ld(_tn(l,"path","rope.sin"),rs);
        ld_ref(_tn(l,"mlp","w1.weight"),w1);ld_ref(_tn(l,"mlp","w2.weight"),w2);ld_ref(_tn(l,"mlp","w3.weight"),w3);
        ld(_tn(l,"ln1","weight"),ln1_w);ld(_tn(l,"ln2","weight"),ln2_w);
        off_dqkv=wg_of(w,_tn(l,"path","w_dqkv.weight"));off_qn=wg_of(w,_tn(l,"path","q_norm.weight"));
        off_uq=wg_of(w,_tn(l,"path","w_uq.weight"));off_uk=wg_of(w,_tn(l,"path","w_uk.weight"));
        off_k2v=wg_of(w,_tn(l,"path","w_k2v.weight"));off_qr=wg_of(w,_tn(l,"path","w_qr.weight"));
        off_kr=wg_of(w,_tn(l,"path","w_kr.weight"));off_proj=wg_of(w,_tn(l,"path","c_proj.weight"));
        off_1=wg_of(w,_tn(l,"mlp","w1.weight"));off_2=wg_of(w,_tn(l,"mlp","w2.weight"));off_3=wg_of(w,_tn(l,"mlp","w3.weight"));
        off_ln1=wg_of(w,_tn(l,"ln1","weight"));off_ln2=wg_of(w,_tn(l,"ln2","weight"));
        off_l1=0;off_cq=d;off_lp=d+dc;off_l2i=d+dc+d;off_l2o=d+dc+d+d;off_gu=d+dc+d+d+d;
        return (w_dqkv&&w_uq&&c_proj&&k_norm_w&&w1&&w2&&w3&&ln1_w&&ln2_w);
    }

    void forward(float*h,ForwardBuffers&b,int B,int T,cudaStream_t s) {
        int n=B*T, oq=n*H*dh, ov=oq+n*Hkv*dh, oqr=ov+n*Hkv*dh, okr=oqr+n*H*dhr;
        cudaMemcpyAsync(b.save+off_l1*n,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        cudaMemcpyAsync(b.a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        launch_pytorch_ln_kernel(b.a,ln1_w,n,d,1e-5f,s);
        cudaMemcpyAsync(b.save+off_lp*n,b.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        launch_linear_dispatch(w_dqkv.data,w_dqkv.qt,b.a,b.m,n,dc,d,s);
        launch_pytorch_ln_kernel(b.m,q_norm_w,n,dc,1e-5f,s);
        cudaMemcpyAsync(b.save+off_cq*n,b.m,n*dc*sizeof(float),cudaMemcpyDeviceToDevice,s);
        launch_linear_dispatch(w_uq.data,w_uq.qt,b.m,b.a,n,H*dh,dc,s);
        launch_linear_dispatch(w_uk.data,w_uk.qt,b.m,b.m+oq,n,Hkv*dh,dc,s);
        launch_linear_dispatch(w_k2v.data,w_k2v.qt,b.m+oq,b.m+ov,n,Hkv*dh,Hkv*dh,s);
        launch_linear_dispatch(w_qr.data,w_qr.qt,b.save+off_lp*n,b.m+oqr,n,H*dhr,d,s);
        launch_linear_dispatch(w_kr.data,w_kr.qt,b.save+off_lp*n,b.m+okr,n,Hkv*dhr,d,s);
        if(rqc)launch_rope_fp32_hf(b.m+oqr,rqc,rqs,B,T,H,dhr,s);
        if(rc)launch_rope_fp32_hf(b.m+okr,rc,rs,B,T,Hkv,dhr,s);
        float*ab=b.m+okr+n*Hkv*dhr,*Qf=ab,*Kf=Qf+n*H*dq,*Vf=Kf+n*H*dq;
        build_qkv_fp32_kernel(b.a,b.m+oq,b.m+ov,b.m+oqr,b.m+okr,Qf,Kf,Vf,B,T,H,Hkv,dh,dhr,dq,s);
        if(b.fm)launch_flashattn_fwd_save_stats(Qf,Kf,Vf,Qf,b.fm,b.fl,B,H,T,dq,dh,s);
        else{extern void launch_flash_attn_fp32(const float*,const float*,const float*,float*,int,int,int,int,int,cudaStream_t);
            launch_flash_attn_fp32(Qf,Kf,Vf,Qf,B,H,T,dq,dh,s);}
        launch_transpose_attn(b.a,Qf,H,T,dh,s);
        launch_linear_dispatch(c_proj.data,c_proj.qt,b.a,b.a,n,d,H*dh,s);
        add_f32_m<<<(n*d+BLK-1)/BLK,BLK,0,s>>>(h,h,b.a,n*d);
        cudaMemcpyAsync(b.save+off_l2i*n,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        cudaMemcpyAsync(b.a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        launch_pytorch_ln_kernel(b.a,ln2_w,n,d,1e-5f,s);
        cudaMemcpyAsync(b.save+off_l2o*n,b.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,s);
        launch_linear_dispatch(w1.data,w1.qt,b.a,b.m,n,hd,d,s);launch_linear_dispatch(w3.data,w3.qt,b.a,b.m+n*hd,n,hd,d,s);
        cudaMemcpyAsync(b.save+off_gu*n,b.m,2*n*hd*sizeof(float),cudaMemcpyDeviceToDevice,s);
        silu_mul_f32_m<<<(n*hd+BLK-1)/BLK,BLK,0,s>>>(b.m,b.m,b.m+n*hd,n*hd);
        launch_linear_dispatch(w2.data,w2.qt,b.m,b.a,n,d,hd,s);
        add_f32_m<<<(n*d+BLK-1)/BLK,BLK,0,s>>>(h,h,b.a,n*d);
    }

    void backward(GradBuffers& g,ForwardBuffers&b,float*wg,int B,int T,cudaStream_t s) {
        int n=B*T,hdh=H*dh,hkv=Hkv*dh,oq=n*hdh,ov=oq+n*hkv,oqr=ov+n*hkv,okr=oqr+n*H*dhr,dq=dh+dhr;
        const float*sv_l1=b.save+off_l1*n,*sv_cq=b.save+off_cq*n,*sv_lp=b.save+off_lp*n;
        const float*sv_l2i=b.save+off_l2i*n,*sv_l2o=b.save+off_l2o*n,*sv_gu=b.save+off_gu*n;
        cublasHandle_t ch=get_cublas_handle();cublasSetStream(ch,s);float a1=1.0f,b0=0.0f,b1=1.0f;
        // MLP bwd
        launch_copy_f32(g.da,g.dh,n*d,s);
        launch_silu_mul_inline(g.dm,sv_gu,sv_gu+n*hd,n*hd,s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,hd,n,d,&a1,w2.f32(),hd,g.da,d,&b0,g.dm,hd);
        if(off_2!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hd,d,n,&a1,g.dm,hd,g.da,d,&b1,wg+off_2,hd);
        launch_silu_mul_bwd_fp32(g.dm,sv_gu,sv_gu+n*hd,g.dm,g.dm+n*hd,n*hd,s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,hd,&a1,w1.f32(),d,g.dm,hd,&b1,g.da,d);
        if(off_1!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,hd,n,&a1,sv_l2o,d,g.dm,hd,&b1,wg+off_1,d);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,hd,&a1,w3.f32(),d,g.dm+n*hd,hd,&b1,g.da,d);
        if(off_3!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,hd,n,&a1,sv_l2o,d,g.dm+n*hd,hd,&b1,wg+off_3,d);
        launch_layernorm_bwd_fp32(g.da,sv_l2i,ln2_w,g.da,0,n,d,s);
        add_f32_m<<<(n*d+BLK-1)/BLK,BLK,0,s>>>(g.dh,g.dh,g.da,n*d);
        // MLA path bwd
        launch_copy_f32(g.da,g.dh,n*d,s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,hdh,n,d,&a1,c_proj.f32(),hdh,g.da,d,&b0,g.dm,hdh);
        if(off_proj!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hdh,d,n,&a1,g.dm,hdh,g.da,d,&b1,wg+off_proj,hdh);
        int ab_off=oqr+n*Hkv*dhr,qf_sz=n*H*dq,vf_sz=n*H*dh;
        float*ab=g.dm+ab_off;launch_transpose_attn(ab,g.dm,H,T,dh,s);
        float*Qf=ab,*Kf=Qf+qf_sz,*Vf=Kf+qf_sz;
        if(b.fm&&b.fl){cudaMemsetAsync(Qf,0,qf_sz*4,s);cudaMemsetAsync(Kf,0,qf_sz*4,s);cudaMemsetAsync(Vf,0,vf_sz*4,s);
            launch_flash_attn_bwd_fp32(Qf,Kf,Vf,Qf,ab,b.fm,b.fl,Qf,Kf,Vf,B,H,T,dq,dh,s);}
        float*dq_=g.dm+ab_off+qf_sz+qf_sz+vf_sz,*dk_=dq_+n*hdh,*dv_=dk_+n*hkv,*dqr_=dv_+n*hkv,*dkr_=dqr_+n*H*dhr;
        build_qkv_bwd_kernel(Qf,Kf,Vf,dq_,dk_,dv_,dqr_,dkr_,B,T,H,Hkv,dh,dhr,dq,s);
        if(rqc)launch_rope_bwd_fp32_hf(dqr_,dqr_,rqc,rqs,B,T,H,dhr,s);
        if(rc)launch_rope_bwd_fp32_hf(dkr_,dkr_,rc,rs,B,T,Hkv,dhr,s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,H*dhr,&a1,w_qr.f32(),d,dqr_,H*dhr,&b1,g.da,d);
        if(off_qr!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,H*dhr,n,&a1,sv_lp,d,dqr_,H*dhr,&b1,wg+off_qr,d);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,Hkv*dhr,&a1,w_kr.f32(),d,dkr_,Hkv*dhr,&b1,g.da,d);
        if(off_kr!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,Hkv*dhr,n,&a1,sv_lp,d,dkr_,Hkv*dhr,&b1,wg+off_kr,d);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,hkv,n,hkv,&a1,w_k2v.f32(),hkv,dv_,hkv,&b1,dk_,hkv);
        if(off_k2v!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hkv,hkv,n,&a1,dv_,hkv,dk_,hkv,&b1,wg+off_k2v,hkv);
        float*d_cq=dkr_+n*Hkv*dhr;cudaMemsetAsync(d_cq,0,n*dc*sizeof(float),s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,dc,n,hkv,&a1,w_uk.f32(),dc,dk_,hkv,&b1,d_cq,dc);
        if(off_uk!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hkv,dc,n,&a1,dk_,hkv,sv_cq,dc,&b1,wg+off_uk,hkv);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,dc,n,hdh,&a1,w_uq.f32(),dc,dq_,hdh,&b1,d_cq,dc);
        if(off_uq!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hdh,dc,n,&a1,dq_,hdh,sv_cq,dc,&b1,wg+off_uq,hdh);
        launch_layernorm_bwd_fp32(d_cq,sv_cq,q_norm_w,d_cq,0,n,dc,s);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,dc,&a1,w_dqkv.f32(),d,d_cq,dc,&b1,g.da,d);
        if(off_dqkv!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,dc,d,n,&a1,d_cq,dc,g.da,d,&b1,wg+off_dqkv,dc);
        launch_layernorm_bwd_fp32(g.da,sv_l1,ln1_w,g.da,0,n,d,s);
        add_f32_m<<<(n*d+BLK-1)/BLK,BLK,0,s>>>(g.dh,g.dh,g.da,n*d);
    }

    int ws_per_token(){return std::max(d*4*2/3/256*256, dc+H*dh+2*Hkv*dh+H*dhr+Hkv*dhr+H*dq*3);}
    int sv_per_token(){return d+dc+d+d+d+2*(d*4*2/3/256*256);}
};

extern "C" {
static bool mla_init(void*s,const ModelConfig&c,const TensorMap&w,int l){return((RinaMLAImpl*)s)->init(c,w,l);}
static void mla_fwd(void*s,float*h,ForwardBuffers&b,int B,int T,cudaStream_t st){((RinaMLAImpl*)s)->forward(h,b,B,T,st);}
static void mla_bwd(void*s,GradBuffers&g,ForwardBuffers&b,float*wg,int B,int T,cudaStream_t st){((RinaMLAImpl*)s)->backward(g,b,wg,B,T,st);}
static int mla_ws(void*s,int d,int h,int hd){return((RinaMLAImpl*)s)->ws_per_token();}
static int mla_sv(void*s,int d,int h,int hd){return((RinaMLAImpl*)s)->sv_per_token();}
static void mla_del(void*s){delete(RinaMLAImpl*)s;}
}
static const LayerVTable mla_vtab={mla_init,mla_fwd,mla_bwd,mla_ws,mla_sv,mla_del};
Layer create_rina_mla_layer(){Layer l;l.impl=new RinaMLAImpl();l.vtab=&mla_vtab;return l;}
