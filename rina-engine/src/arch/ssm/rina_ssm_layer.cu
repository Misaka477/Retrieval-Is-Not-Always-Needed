#include "core/layer.h"
#include "core/config.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "kernels/gemm.cuh"
#include <cstdio>
#include <cmath>
#include <string>

extern void launch_linear_fp32(const float*, const float*, float*, int, int, int, cudaStream_t);
extern "C" void launch_pytorch_ln_kernel(float*, const float*, int, int, float, cudaStream_t);
extern void launch_sigmoid_fp32(float*, int, cudaStream_t);
extern void launch_ssm_agg_fp32(const float*, const float*, const float*,
    const float*, const float*, const float*, float*, float*,
    int, int, int, cudaStream_t);
extern void launch_ssm_scan_fp32(const float*, const float*, float*,
    int, int, int, int, cudaStream_t);
extern void launch_ssm_scan_bwd_fp32(const float*, const float*,
    const float*, const float*, float*, float*,
    int, int, int, int, cudaStream_t);
extern void launch_ssm_agg_bwd_fp32(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    float*, float*, float*, float*, float*, float*,
    int, int, int, cudaStream_t);
extern void launch_sigmoid_bwd_fp32(float*, const float*, int, cudaStream_t);
extern void launch_layernorm_bwd_fp32(const float*, const float*,
    const float*, float*, float*, int, int, cudaStream_t);
extern void launch_silu_mul_bwd_fp32(const float*, const float*, const float*,
    float*, float*, int, cudaStream_t);
extern void launch_silu_mul_inline(float*, const float*, const float*, int, cudaStream_t);
extern void launch_copy_f32(float*, const float*, int, cudaStream_t);

static const int BLK = 256;
__global__ void add_f32_s(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x*BLK + threadIdx.x; if (i < n) c[i] = a[i] + b[i];
}
__global__ void silu_mul_f32_s(float* o, const float* g, const float* u, int n) {
    int i = blockIdx.x*BLK + threadIdx.x;
    if (i < n) o[i] = (g[i]/(1.0f+expf(-g[i])))*u[i];
}

static std::string _tn(int l, const char* c, const char* p) {
    char b[128]; snprintf(b,128,"transformer.h.%d.%s.%s",l,c,p); return b;
}

static size_t wg_offset_of(const TensorMap& w, const std::string& name) {
    size_t off=0; for(auto&[n_,wt]:w.tensors){
        if(wt.quant_type!=QuantType::FP32)continue;
        if(n_==name)return off; off+=wt.n_elems;}
    return (size_t)-1;
}

struct RinaSSMImpl {
    int d, H, Hkv, dh, dc, hd, ssm_steps;
    const float *w_dq, *q_norm_w, *w_mem[3], *w_decay[3], *w_out;
    const float *w1, *w2, *w3, *ln1_w, *ln2_w;
    size_t off_dq, off_qn, off_mem[3], off_decay[3], off_out;
    size_t off_1, off_2, off_3, off_ln1, off_ln2;
    int off_ln1_out, off_cq, off_db, off_mems, off_ds, off_ln2_in, off_ln2_out, off_gu;

    bool init(const ModelConfig& cfg, const TensorMap& weights, int l) {
        d=cfg.dim; H=cfg.n_heads; Hkv=cfg.n_kv_heads; dh=cfg.head_dim;
        dc=cfg.d_c?cfg.d_c:160; ssm_steps=cfg.ssm_steps; hd=d*4*2/3/256*256;
        auto ld=[&](const std::string& name, const float*& ptr){
            auto*t=weights.get(name); if(!t)return false; ptr=(const float*)t->data; return true;};
        ld(_tn(l,"path","w_dq.weight"),w_dq); ld(_tn(l,"path","q_norm.weight"),q_norm_w);
        for(int k=0;k<ssm_steps;k++){
            char b[128]; snprintf(b,128,"transformer.h.%d.path.w_mem.%d.weight",l,k); ld(b,w_mem[k]);
            snprintf(b,128,"transformer.h.%d.path.w_decay.%d.weight",l,k); ld(b,w_decay[k]);}
        ld(_tn(l,"path","w_out.weight"),w_out);
        ld(_tn(l,"mlp","w1.weight"),w1); ld(_tn(l,"mlp","w2.weight"),w2); ld(_tn(l,"mlp","w3.weight"),w3);
        ld(_tn(l,"ln1","weight"),ln1_w); ld(_tn(l,"ln2","weight"),ln2_w);
        off_dq=wg_offset_of(weights,_tn(l,"path","w_dq.weight")); off_qn=wg_offset_of(weights,_tn(l,"path","q_norm.weight"));
        for(int k=0;k<ssm_steps;k++){
            char b[128]; snprintf(b,128,"transformer.h.%d.path.w_mem.%d.weight",l,k); off_mem[k]=wg_offset_of(weights,b);
            snprintf(b,128,"transformer.h.%d.path.w_decay.%d.weight",l,k); off_decay[k]=wg_offset_of(weights,b);}
        off_out=wg_offset_of(weights,_tn(l,"path","w_out.weight"));
        off_1=wg_offset_of(weights,_tn(l,"mlp","w1.weight")); off_2=wg_offset_of(weights,_tn(l,"mlp","w2.weight"));
        off_3=wg_offset_of(weights,_tn(l,"mlp","w3.weight")); off_ln1=wg_offset_of(weights,_tn(l,"ln1","weight"));
        off_ln2=wg_offset_of(weights,_tn(l,"ln2","weight"));
        off_ln1_out=0; off_cq=d; off_db=d+dc; off_mems=off_db+ssm_steps*H;
        off_ds=off_mems+3*H*dh; off_ln2_in=off_ds+3*H; off_ln2_out=off_ln2_in+d; off_gu=off_ln2_out+d;
        return (w_dq && q_norm_w && w_out && w1 && w2 && w3 && ln1_w && ln2_w);
    }

    void forward(float* h, ForwardBuffers& bufs, int B, int T, cudaStream_t stream) {
        int n=B*T, ncq=n*dc, ms=n*H*dh, doff=ncq+ssm_steps*ms, d2=d+H*dh;
        cudaMemcpyAsync(bufs.save+off_ln1_out*n,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(bufs.a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_pytorch_ln_kernel(bufs.a,ln1_w,n,d,1e-5f,stream);
        launch_linear_fp32(bufs.a,w_dq,bufs.m,n,dc,d,stream);
        launch_pytorch_ln_kernel(bufs.m,q_norm_w,n,dc,1e-5f,stream);
        cudaMemcpyAsync(bufs.save+off_cq*n,bufs.m,n*dc*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        for(int k=0;k<ssm_steps;k++){
            launch_linear_fp32(bufs.m,w_mem[k],bufs.m+ncq+k*ms,n,H*dh,dc,stream);
            launch_linear_fp32(bufs.m,w_decay[k],bufs.m+doff+k*n*H,n,H,dc,stream);
            cudaMemcpyAsync(bufs.save+(off_db+k*H)*n,bufs.m+doff+k*n*H,n*H*sizeof(float),cudaMemcpyDeviceToDevice,stream);
            launch_sigmoid_fp32(bufs.m+doff+k*n*H,n*H,stream);}
        float* da=bufs.m+doff;
        float* ma=bufs.m+doff+3*n*H;
        launch_ssm_agg_fp32(bufs.m+ncq,bufs.m+ncq+ms,bufs.m+ncq+2*ms,
                            da,da+n*H,da+2*n*H,da,ma,H,dh,n,stream);
        cudaMemcpyAsync(bufs.save+off_mems*n,bufs.m+ncq,3*ms*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(bufs.save+off_ds*n,da,3*n*H*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_ssm_scan_fp32(ma,da,ma,B,T,H,dh,stream);
        for(int i=0;i<n;i++){
            cudaMemcpyAsync(bufs.m+i*d2,bufs.a+i*d,d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
            cudaMemcpyAsync(bufs.m+i*d2+d,ma+i*H*dh,H*dh*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        }
        launch_linear_fp32(bufs.m,w_out,bufs.a,n,d,d2,stream);
        add_f32_s<<<(n*d+BLK-1)/BLK,BLK,0,stream>>>(h,h,bufs.a,n*d);
        cudaMemcpyAsync(bufs.save+off_ln2_in*n,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(bufs.a,h,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_pytorch_ln_kernel(bufs.a,ln2_w,n,d,1e-5f,stream);
        cudaMemcpyAsync(bufs.save+off_ln2_out*n,bufs.a,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_linear_fp32(bufs.a,w1,bufs.m,n,hd,d,stream);
        launch_linear_fp32(bufs.a,w3,bufs.m+n*hd,n,hd,d,stream);
        cudaMemcpyAsync(bufs.save+off_gu*n,bufs.m,2*n*hd*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        silu_mul_f32_s<<<(n*hd+BLK-1)/BLK,BLK,0,stream>>>(bufs.m,bufs.m,bufs.m+n*hd,n*hd);
        launch_linear_fp32(bufs.m,w2,bufs.a,n,d,hd,stream);
        add_f32_s<<<(n*d+BLK-1)/BLK,BLK,0,stream>>>(h,h,bufs.a,n*d);
    }

    void backward(GradBuffers& grad, ForwardBuffers& bufs, float* wg, int B, int T, cudaStream_t stream) {
        int n=B*T, ncq=n*dc, ms=n*H*dh, doff=ncq+ssm_steps*ms, d2=d+H*dh;
        cublasHandle_t ch=get_cublas_handle(); cublasSetStream(ch,stream);
        float a1=1.0f,b0=0.0f,b1=1.0f;
        const float* sv_l1=bufs.save+off_ln1_out*n, *sv_cq=bufs.save+off_cq*n;
        const float* sv_dr=bufs.save+off_db*n, *sv_mems=bufs.save+off_mems*n;
        const float* sv_ds=bufs.save+off_ds*n, *sv_l2i=bufs.save+off_ln2_in*n;
        const float* sv_l2o=bufs.save+off_ln2_out*n, *sv_gu=bufs.save+off_gu*n;
        // MLP bwd
        launch_copy_f32(grad.da,grad.dh,n*d,stream);
        launch_silu_mul_inline(grad.dm,sv_gu,sv_gu+n*hd,n*hd,stream);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,hd,n,d,&a1,w2,hd,grad.da,d,&b0,grad.dm,hd);
        if(off_2!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,hd,d,n,&a1,grad.dm,hd,grad.da,d,&b1,wg+off_2,hd);
        launch_silu_mul_bwd_fp32(grad.dm,sv_gu,sv_gu+n*hd,grad.dm,grad.dm+n*hd,n*hd,stream);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,hd,&a1,w1,d,grad.dm,hd,&b1,grad.da,d);
        if(off_1!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,hd,n,&a1,sv_l2o,d,grad.dm,hd,&b1,wg+off_1,d);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,hd,&a1,w3,d,grad.dm+n*hd,hd,&b1,grad.da,d);
        if(off_3!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d,hd,n,&a1,sv_l2o,d,grad.dm+n*hd,hd,&b1,wg+off_3,d);
        launch_layernorm_bwd_fp32(grad.da,sv_l2i,ln2_w,grad.da,0,n,d,stream);
        add_f32_s<<<(n*d+BLK-1)/BLK,BLK,0,stream>>>(grad.dh,grad.dh,grad.da,n*d);
        // SSM path bwd
        launch_copy_f32(grad.da,grad.dh,n*d,stream);
        cudaMemcpyAsync(grad.dm,sv_l1,n*d*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_pytorch_ln_kernel(grad.dm,ln1_w,n,d,1e-5f,stream);
        launch_ssm_agg_fp32(sv_mems,sv_mems+ms,sv_mems+2*ms,sv_ds,sv_ds+n*H,sv_ds+2*n*H,
            grad.dm+n*d2,grad.dm+n*d2+n*H,H,dh,n,stream);
        float* mem_agg=grad.dm+n*d2+n*H, *dec_agg=grad.dm+n*d2;
        float* mem_save=grad.dm+n*d2+n*(H*dh+H);
        cudaMemcpyAsync(mem_save,mem_agg,n*H*dh*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        launch_ssm_scan_fp32(mem_agg,dec_agg,mem_agg,B,T,H,dh,stream);
        cudaMemcpyAsync(grad.dm+n*d,mem_agg,n*H*dh*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d2,n,d,&a1,w_out,d2,grad.da,d,&b0,grad.dm+n*d2,d2);
        if(off_out!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,d2,d,n,&a1,grad.dm,d2,grad.da,d,&b1,wg+off_out,d2);
        const float* d_sf=grad.dm+n*d2+n*d;
        float* d_mem=grad.dm+n*d2, *d_dec=d_mem+n*H*dh;
        launch_ssm_scan_bwd_fp32(d_sf,mem_save,dec_agg,mem_agg,d_mem,d_dec,B,T,H,dh,stream);
        float* d_mems[3]={d_mem,d_mem+n*H*dh,d_mem+2*n*H*dh};
        float* d_decs[3]={d_dec,d_dec+n*H,d_dec+2*n*H};
        launch_ssm_agg_bwd_fp32(d_mem,d_dec,sv_mems,sv_mems+ms,sv_mems+2*ms,
            sv_ds,sv_ds+n*H,sv_ds+2*n*H,d_mems[0],d_mems[1],d_mems[2],
            d_decs[0],d_decs[1],d_decs[2],H,dh,n,stream);
        float* d_cq=grad.dm; cudaMemsetAsync(d_cq,0,n*dc*sizeof(float),stream);
        for(int k=0;k<ssm_steps;k++){
            launch_sigmoid_bwd_fp32(d_decs[k],sv_dr+k*n*H,n*H,stream);
            if(off_decay[k]!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,H,dc,n,&a1,d_decs[k],H,sv_cq,dc,&b1,wg+off_decay[k],H);
            cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,dc,n,H,&a1,w_decay[k],dc,d_decs[k],H,&b1,d_cq,dc);
            if(off_mem[k]!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,H*dh,dc,n,&a1,d_mems[k],H*dh,sv_cq,dc,&b1,wg+off_mem[k],H*dh);
            cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,dc,n,H*dh,&a1,w_mem[k],dc,d_mems[k],H*dh,&b1,d_cq,dc);}
        launch_layernorm_bwd_fp32(d_cq,sv_cq,q_norm_w,d_cq,0,n,dc,stream);
        cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_N,d,n,dc,&a1,w_dq,d,d_cq,dc,&b1,grad.da,d);
        if(off_dq!=(size_t)-1)cublasSgemm(ch,CUBLAS_OP_N,CUBLAS_OP_T,dc,d,n,&a1,d_cq,dc,grad.da,d,&b1,wg+off_dq,dc);
        launch_layernorm_bwd_fp32(grad.da,sv_l1,ln1_w,grad.da,0,n,d,stream);
        add_f32_s<<<(n*d+BLK-1)/BLK,BLK,0,stream>>>(grad.dh,grad.dh,grad.da,n*d);
    }

    int workspace_per_token() {
        int w=dc+3*H*dh+ssm_steps*H+H+H*dh+d+H*dh+3*H+H*dh*2;
        return std::max(d*4*2/3/256*256,w);
    }
    int saved_per_token() {
        return d+dc+ssm_steps*H+3*H*dh+3*H+d+d+2*(d*4*2/3/256*256);
    }
};

extern "C" {
static bool ssm_init(void* s,const ModelConfig& c,const TensorMap& w,int l){return ((RinaSSMImpl*)s)->init(c,w,l);}
static void ssm_fwd(void* s,float* h,ForwardBuffers& b,int B,int T,cudaStream_t st){((RinaSSMImpl*)s)->forward(h,b,B,T,st);}
static void ssm_bwd(void* s,GradBuffers& g,ForwardBuffers& b,float* wg,int B,int T,cudaStream_t st){((RinaSSMImpl*)s)->backward(g,b,wg,B,T,st);}
static int ssm_ws(void* s,int d,int h,int hd){return ((RinaSSMImpl*)s)->workspace_per_token();}
static int ssm_sv(void* s,int d,int h,int hd){return ((RinaSSMImpl*)s)->saved_per_token();}
static void ssm_del(void* s){delete(RinaSSMImpl*)s;}
}
static const LayerVTable ssm_vtab={ssm_init,ssm_fwd,ssm_bwd,ssm_ws,ssm_sv,ssm_del};
Layer create_rina_ssm_layer(){Layer l;l.impl=new RinaSSMImpl();l.vtab=&ssm_vtab;return l;}
