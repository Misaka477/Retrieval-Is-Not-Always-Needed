"""
Attractor: fused attractor-only forward + backward for MoHE experts.
Forward: loads pre-computed h_fast -> field -> field_mix -> LN -> slow_gate -> h_out.
Backward: CUDA kernel for attractor param grads, recomputes intermediates in PyTorch.
"""
import os, glob, torch
_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
from torch.utils.cpp_extension import load_inline

_K3_ATTRACTOR = None

def _find_msvc():
    for root in [r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"D:\Software_Development\Microsoft Visual Studio"]:
        hits = glob.glob(os.path.join(root, r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
        if hits: return os.path.dirname(hits[0])
    return None

def _load():
    global _K3_ATTRACTOR
    if _K3_ATTRACTOR is not None: return
    _msvc = _find_msvc()
    if _msvc: os.environ["PATH"] = _msvc + os.pathsep + os.environ.get("PATH", "")
    _cu = os.environ.get("CUDA_PATH", r"D:\Software_Development\CUDA_Toolkit_12.4")
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")

    C_FWD = r"""
__global__ void k3_light_fwd(float* h_out, float* h_fast_out,
    const float* h_fast, const float* h, const float* x,
    const float* P, const float* fmw, const float* fmb,
    const float* nw, const float* nb, const float* sw, const float* sb,
    int bs, int dm, int ne){
    int b=blockIdx.x,tid=threadIdx.x;
    if(b>=bs)return;
    extern __shared__ float s[];
    float* s_hf=s; float* s_fld=s+dm; float* s_tmp=s+2*dm; float* s_red=s+3*dm;
    for(int e=0;e<ne;e++){
        int eb=e*dm, ed=e*dm*dm;
        for(int d=tid;d<dm;d+=blockDim.x)s_hf[d]=h_fast[e*bs*dm+b*dm+d];
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x){
            float fd=0; for(int j=0;j<dm;j++)fd=__fma_rn(s_hf[j],P[ed+j*dm+d],fd);
            s_fld[d]=fd;}
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x){
            float fm_val=fmb[eb+d]; for(int j=0;j<dm;j++)fm_val=__fma_rn(s_fld[j],fmw[ed+d*dm+j],fm_val);
            s_tmp[d]=fm_val;}
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x)s_fld[d]=s_tmp[d];
        __syncthreads();
        float mn_p=0; for(int d=tid;d<dm;d+=blockDim.x)mn_p+=s_fld[d];
        s_red[tid]=mn_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float mn=s_red[0]/dm, vr_p=0;
        for(int d=tid;d<dm;d+=blockDim.x){float dv=s_fld[d]-mn;vr_p+=dv*dv;}
        s_red[tid]=vr_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float rstd=__frsqrt_rn(s_red[0]/dm+1e-5f);
        for(int d=tid;d<dm;d+=blockDim.x)s_tmp[d]=(s_fld[d]-mn)*rstd*nw[eb+d]+nb[eb+d];
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x)s_fld[d]=s_tmp[d];
        __syncthreads();
        float sg_p=0;
        for(int d=tid;d<dm;d+=blockDim.x)sg_p+=h[b*dm+d]*sw[e*2*dm+d]+x[b*dm+d]*sw[e*2*dm+dm+d];
        s_red[tid]=sg_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float gate=__frcp_rn(1.f+expf(-fminf(fmaxf(s_red[0]+sb[e],-30.f),30.f)));
        for(int d=tid;d<dm;d+=blockDim.x){
            int base=e*bs*dm+b*dm+d;
            h_out[base]=__fma_rn(gate,s_fld[d],s_hf[d]*0.5f); h_fast_out[base]=s_hf[d];}
        __syncthreads();}}
void fwd(float* h_out,float* h_fast_out,
    const float* h_fast,const float* h,const float* x,
    const float* P,const float* fmw,const float* fmb,
    const float* nw,const float* nb,const float* sw,const float* sb,
    int bs,int dm,int ne){
    k3_light_fwd<<<bs,256,(3*dm+256)*sizeof(float)>>>(h_out,h_fast_out,
        h_fast,h,x,P,fmw,fmb,nw,nb,sw,sb,bs,dm,ne);}
"""
    B_FWD = ("void fwd(float*,float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "int,int,int);"
             "void f_fwd(torch::Tensor h_out,torch::Tensor h_fast_out,"
             "torch::Tensor h_fast,torch::Tensor h,torch::Tensor x,"
             "torch::Tensor P,torch::Tensor fmw,torch::Tensor fmb,"
             "torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
             "fwd(h_out.data_ptr<float>(),h_fast_out.data_ptr<float>(),"
             "h_fast.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
             "P.data_ptr<float>(),fmw.data_ptr<float>(),fmb.data_ptr<float>(),"
             "nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
             "h.size(0),h.size(1),P.size(0));}")

    C_BWD = r"""
__device__ void tr(volatile float* s, int t){
    for(int st=blockDim.x/2;st>0;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}}
__global__ void k3_light_bwd(
    float* grad_out,
    const float* g_ho, const float* g_hf, const float* h, const float* x,
    const float* h_fast, const float* fm, const float* nm,
    const float* a_g,
    const float* P, const float* fmw, const float* fmb,
    const float* nw, const float* nb,
    const float* sw, const float* sb,
    int bs, int dm, int ne){
    int e=blockIdx.x,b=blockIdx.y,tid=threadIdx.x;
    if(e>=ne||b>=bs)return;
    extern __shared__ volatile float s_red[];
    int base=e*bs*dm+b*dm; int eb=e*dm, ed=e*dm*dm;
    float gate=a_g[e*bs+b];
    int off_hc = 0;
    int off_fo = off_hc + ne*bs*dm;
    int off_ls = off_fo + ne*bs*dm;
    int off_nw = off_ls + ne*bs;
    int off_nb = off_nw + ne*dm;
    int off_Po = off_nb + ne*dm;
    int off_ft = off_Po + ne*dm*dm;
    float* g_hc = grad_out + off_hc;
    float* g_fo = grad_out + off_fo;
    float* g_ls = grad_out + off_ls;
    float* g_nw = grad_out + off_nw;
    float* g_nb = grad_out + off_nb;
    float* g_Po = grad_out + off_Po;
    float* g_ft = grad_out + off_ft;

    float gs=0; for(int d=tid;d<dm;d+=blockDim.x)gs+=g_ho[base+d]*nm[base+d];
    s_red[tid]=gs;__syncthreads();tr(s_red,tid);
    float gls=s_red[0]*gate*(1.f-gate); if(tid==0)g_ls[e*bs+b]=gls;

    float mp=0; for(int d=tid;d<dm;d+=blockDim.x)mp+=fm[base+d];
    s_red[tid]=mp;__syncthreads();tr(s_red,tid);float mu=s_red[0]/dm;
    float vp=0; for(int d=tid;d<dm;d+=blockDim.x){float dv=fm[base+d]-mu;vp+=dv*dv;}
    s_red[tid]=vp;__syncthreads();tr(s_red,tid);float rstd=rsqrtf(s_red[0]/dm+1e-5f);
    float sp=0,snp=0;
    for(int d=tid;d<dm;d+=blockDim.x){
        float gn=g_ho[base+d]*gate,nv=(fm[base+d]-mu)*rstd;
        sp+=gn*nw[eb+d];snp+=gn*nw[eb+d]*nv;}
    s_red[tid]=sp;__syncthreads();tr(s_red,tid);float sdy=s_red[0];
    s_red[tid]=snp;__syncthreads();tr(s_red,tid);float sdn=s_red[0];
    for(int d=tid;d<dm;d+=blockDim.x){
        float gn=g_ho[base+d]*gate,nv=(fm[base+d]-mu)*rstd;
        g_fo[base+d]=gn*nw[eb+d]*rstd-rstd/dm*sdy-nv*rstd/dm*sdn;
        g_nw[eb+d]=gn*nv; g_nb[eb+d]=gn;}
    __syncthreads();

    for(int d=tid;d<dm;d+=blockDim.x){
        float gf=0; for(int j=0;j<dm;j++)gf+=g_fo[base+j]*fmw[ed+j*dm+d];g_ft[base+d]=gf;}
    __syncthreads();

    for(int d=tid;d<dm;d+=blockDim.x){
        float ghf=0; for(int j=0;j<dm;j++)ghf+=g_ft[base+j]*P[ed+j*dm+d];
        float gt=__fma_rn(g_ho[base+d],0.5f,ghf)+g_hf[base+d];
        g_hc[base+d]=gt;
        for(int j=0;j<dm;j++)atomicAdd(&g_Po[ed+d*dm+j],h_fast[base+d]*g_ft[base+j]);}
}
void launch_bwd(float* grad_out,
    const float* g_ho,const float* g_hf,const float* h,const float* x,
    const float* h_fast,const float* fm,const float* nm,
    const float* a_g,
    const float* P,const float* fmw,const float* fmb,
    const float* nw,const float* nb,
    const float* sw,const float* sb,
    int bs,int dm,int ne){
    k3_light_bwd<<<dim3(ne,bs),256,0>>>(grad_out,
        g_ho,g_hf,h,x,h_fast,fm,nm,a_g,P,fmw,fmb,nw,nb,sw,sb,bs,dm,ne);}"""
    B_BWD = ("void launch_bwd(float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "int,int,int);"
             "void f_bwd(torch::Tensor grad_out,"
             "torch::Tensor g_ho,torch::Tensor g_hf,torch::Tensor h,torch::Tensor x,"
             "torch::Tensor h_fast,torch::Tensor fm,torch::Tensor nm,torch::Tensor a_g,"
             "torch::Tensor P,torch::Tensor fmw,torch::Tensor fmb,"
             "torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
             "launch_bwd(grad_out.data_ptr<float>(),"
             "g_ho.data_ptr<float>(),g_hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
             "h_fast.data_ptr<float>(),fm.data_ptr<float>(),nm.data_ptr<float>(),a_g.data_ptr<float>(),"
             "P.data_ptr<float>(),fmw.data_ptr<float>(),fmb.data_ptr<float>(),"
             "nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
             "h.size(0),h.size(1),P.size(0));}")

    _K3_ATTRACTOR = load_inline("k3_attractor", 
        cpp_sources=B_FWD + B_BWD, cuda_sources=C_FWD + C_BWD,
        functions=["f_fwd", "f_bwd"], verbose=False)
    print("Attractor kernel ready (fwd + bwd)")


class FusedLightFunction(torch.autograd.Function):
    """Attractor-only fused forward+backward for MoHE experts.

    Forward: (h_fast, h, x_emb, P, fmw, fmb, nw, nb, sw, sb) -> h_out, h_fast_out
    Backward: recomputes intermediates via PyTorch, calls fused CUDA backward.
    """

    @staticmethod
    def forward(ctx, h_fast, h, x_emb, P, fmw, fmb, nw, nb, sw, sb):
        _load()
        ne, bs, dm = P.shape[0], h.shape[0], h.shape[1]
        h_out = torch.empty(ne, bs, dm, device=h.device, dtype=torch.float32)
        h_fast_out = torch.empty(ne, bs, dm, device=h.device, dtype=torch.float32)
        _K3_ATTRACTOR.f_fwd(h_out, h_fast_out, h_fast, h, x_emb,
                              P, fmw, fmb, nw, nb, sw, sb)
        ctx.save_for_backward(h_fast, h, x_emb, P, fmw, fmb, nw, nb, sw, sb)
        return h_out, h_fast_out

    @staticmethod
    def backward(ctx, grad_h_out, grad_h_fast_out):
        h_fast, h, x_emb, P, fmw, fmb, nw, nb, sw, sb = ctx.saved_tensors
        ne, bs, dm = P.shape[0], h.shape[0], h.shape[1]
        if grad_h_out is None:
            grad_h_out = torch.zeros(ne, bs, dm, device=h.device)
        if grad_h_fast_out is None:
            grad_h_fast_out = torch.zeros(ne, bs, dm, device=h.device)

        field = torch.einsum('ebk,ekd->ebd', h_fast, P)
        fm_out = torch.einsum('ebj,edj->ebd', field, fmw) + fmb.unsqueeze(1)
        mu = fm_out.mean(dim=-1, keepdim=True)
        var = fm_out.var(dim=-1, keepdim=True, unbiased=False)
        normed_out = (fm_out - mu) * torch.rsqrt(var + 1e-5) * nw.unsqueeze(1) + nb.unsqueeze(1)
        gate_logit = (torch.einsum('bd,ed->eb', h, sw[:, :dm]) +
                      torch.einsum('bd,ed->eb', x_emb, sw[:, dm:]) + sb.unsqueeze(1))
        a_g = torch.sigmoid(gate_logit)

        off_fo = ne * bs * dm
        off_ls = off_fo + ne * bs * dm
        off_nw = off_ls + ne * bs
        off_nb = off_nw + ne * dm
        off_Po = off_nb + ne * dm
        off_ft = off_Po + ne * dm * dm
        total = off_ft + ne * bs * dm

        grad_out = torch.zeros(total, device=h.device, dtype=torch.float32)

        _K3_ATTRACTOR.f_bwd(grad_out,
                              grad_h_out.contiguous(), grad_h_fast_out.contiguous(),
                              h.contiguous(), x_emb.contiguous(),
                              h_fast.contiguous(),
                              fm_out.contiguous(), normed_out.contiguous(), a_g.contiguous(),
                              P.contiguous(), fmw.contiguous(), fmb.contiguous(),
                              nw.contiguous(), nb.contiguous(), sw.contiguous(), sb.contiguous())

        g_hc = grad_out[:off_fo].view(ne, bs, dm)
        g_fo = grad_out[off_fo:off_ls].view(ne, bs, dm)
        g_ls = grad_out[off_ls:off_nw].view(ne, bs)
        g_nw = grad_out[off_nw:off_nb].view(ne, dm)
        g_nb = grad_out[off_nb:off_Po].view(ne, dm)
        g_Po = grad_out[off_Po:off_ft].view(ne, dm, dm)

        grad_h_fast = g_hc
        grad_h = torch.einsum('eb,ed->bd', g_ls, sw[:, :dm])
        grad_x_emb = torch.einsum('eb,ed->bd', g_ls, sw[:, dm:])
        grad_P = g_Po
        grad_fmw = torch.einsum('ebd,ebj->edj', g_fo, field)
        grad_fmb = g_fo.sum(dim=1)
        grad_nw = g_nw
        grad_nb = g_nb
        grad_sw_h = torch.einsum('eb,bd->ed', g_ls, h)
        grad_sw_x = torch.einsum('eb,bd->ed', g_ls, x_emb)
        grad_sw = torch.cat([grad_sw_h, grad_sw_x], dim=-1)
        grad_sb = g_ls.sum(dim=1)

        return (grad_h_fast, grad_h, grad_x_emb, grad_P,
                grad_fmw, grad_fmb, grad_nw, grad_nb, grad_sw, grad_sb)
