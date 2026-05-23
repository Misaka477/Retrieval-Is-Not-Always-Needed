"""K3 training support: fused forward + CUDA backward + autograd Function."""
import os, glob, torch
from torch.utils.cpp_extension import load_inline

_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
_K3_FWD = None
_K3_BWD = None

def _find_msvc():
    for root in [r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"D:\Software_Development\Microsoft Visual Studio"]:
        hits = glob.glob(os.path.join(root, r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
        if hits: return os.path.dirname(hits[0])
    return None

def _setup_env():
    _msvc = _find_msvc()
    if _msvc: os.environ["PATH"] = _msvc + os.pathsep + os.environ.get("PATH", "")
    _cu = os.environ.get("CUDA_PATH", r"D:\Software_Development\CUDA_Toolkit_12.4")
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")

def _load_fwd():
    global _K3_FWD
    if _K3_FWD is not None: return
    _setup_env()
    C = r"""
__global__ void k3_fwd(float* h_out, float* h_fast_out,
    float* sa_a, float* sa_b, float* sa_xp,
    float* sa_field, float* sa_fm_out, float* sa_normed, float* sa_gate,
    const float* h, const float* x,
    const float* gw, const float* gb,
    const float* fw, const float* fb,
    const float* pw, const float* pb,
    const float* P,
    const float* fmw, const float* fmb,
    const float* nw, const float* nb,
    const float* sw, const float* sb,
    int bs, int dm, int ne){
    int b=blockIdx.x,tid=threadIdx.x;
    if(b>=bs)return;
    extern __shared__ float s[];
    float* s_hf=s; float* s_fld=s+dm; float* s_red=s+2*dm;
    for(int e=0;e<ne;e++){
        int eb=e*dm, e2d=e*2*dm*dm, ed=e*dm*dm;
        for(int d=tid;d<dm;d+=blockDim.x){
            float sa=gb[eb+d],sbv=fb[eb+d],xp=pb[eb+d];
            for(int k=0;k<dm;k++){
                float hk=h[b*dm+k],xk=x[b*dm+k];
                sa=__fma_rn(hk,gw[e2d+d*2*dm+k],sa); sa=__fma_rn(xk,gw[e2d+d*2*dm+dm+k],sa);
                sbv=__fma_rn(hk,fw[e2d+d*2*dm+k],sbv); sbv=__fma_rn(xk,fw[e2d+d*2*dm+dm+k],sbv);
                xp=__fma_rn(xk,pw[ed+d*dm+k],xp);}
            float sig_a=__frcp_rn(1.f+expf(-fminf(fmaxf(sa,-30.f),30.f)));
            float sig_b=__frcp_rn(1.f+expf(-fminf(fmaxf(sbv,-30.f),30.f)));
            int ebo=e*bs*dm+b*dm+d; sa_a[ebo]=sig_a; sa_b[ebo]=sig_b; sa_xp[ebo]=xp;
            s_hf[d]=__fma_rn(sig_b,xp,sig_a*h[b*dm+d]);}
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x){
            float fd=0; for(int j=0;j<dm;j++)fd=__fma_rn(s_hf[j],P[ed+j*dm+d],fd);
            s_fld[d]=fd; sa_field[e*bs*dm+b*dm+d]=fd;}
        __syncthreads();
        for(int d=tid;d<dm;d+=blockDim.x){
            float fm_val=fmb[eb+d]; for(int j=0;j<dm;j++)fm_val=__fma_rn(s_fld[j],fmw[ed+d*dm+j],fm_val);
            s_fld[d]=fm_val; sa_fm_out[e*bs*dm+b*dm+d]=fm_val;}
        __syncthreads();
        float mn_p=0; for(int d=tid;d<dm;d+=blockDim.x)mn_p+=s_fld[d];
        s_red[tid]=mn_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float mn=s_red[0]/dm, vr_p=0;
        for(int d=tid;d<dm;d+=blockDim.x){float dv=s_fld[d]-mn;vr_p+=dv*dv;}
        s_red[tid]=vr_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float rstd=__frsqrt_rn(s_red[0]/dm+1e-5f);
        for(int d=tid;d<dm;d+=blockDim.x)s_fld[d]=(s_fld[d]-mn)*rstd*nw[eb+d]+nb[eb+d];
        __syncthreads();
        float sg_p=sb[e];
        for(int d=tid;d<dm;d+=blockDim.x)sg_p+=h[b*dm+d]*sw[e*2*dm+d]+x[b*dm+d]*sw[e*2*dm+dm+d];
        s_red[tid]=sg_p;__syncthreads();
        for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s_red[tid]+=s_red[tid+st];__syncthreads();}
        float gate=__frcp_rn(1.f+expf(-fminf(fmaxf(s_red[0],-30.f),30.f)));
        if(tid==0) sa_gate[e*bs+b]=gate;
        for(int d=tid;d<dm;d+=blockDim.x){
            int ebo=e*bs*dm+b*dm+d;
            h_out[ebo]=__fma_rn(gate*0.1f,s_fld[d],s_hf[d]); h_fast_out[ebo]=s_hf[d];}
        __syncthreads();}}
void l3_fwd(float* h_out,float* h_fast_out,float* sa_a,float* sa_b,float* sa_xp,
    float* sa_field,float* sa_fm_out,float* sa_normed,float* sa_gate,
    const float* h,const float* x,const float* gw,const float* gb,
    const float* fw,const float* fb,const float* pw,const float* pb,const float* P,
    const float* fmw,const float* fmb,const float* nw,const float* nb,const float* sw,const float* sb,
    int bs,int dm,int ne){
    k3_fwd<<<bs,256,(2*dm+256)*sizeof(float)>>>(h_out,h_fast_out,
        sa_a,sa_b,sa_xp,sa_field,sa_fm_out,sa_normed,sa_gate,
        h,x,gw,gb,fw,fb,pw,pb,P,fmw,fmb,nw,nb,sw,sb,bs,dm,ne);}
"""
    B = ("void l3_fwd(float*,float*,float*,float*,float*,float*,float*,float*,float*,"
         "const float*,const float*,const float*,const float*,const float*,const float*,"
         "const float*,const float*,const float*,const float*,const float*,const float*,"
         "const float*,const float*,const float*,"
         "int,int,int);"
         "void fwd(torch::Tensor h_out,torch::Tensor h_fast_out,"
         "torch::Tensor sa_a,torch::Tensor sa_b,torch::Tensor sa_xp,"
         "torch::Tensor sa_field,torch::Tensor sa_fm_out,torch::Tensor sa_normed,torch::Tensor sa_gate,"
         "torch::Tensor h,torch::Tensor x,"
         "torch::Tensor gw,torch::Tensor gb,torch::Tensor fw,torch::Tensor fb,"
         "torch::Tensor pw,torch::Tensor pb,torch::Tensor P,"
         "torch::Tensor fmw,torch::Tensor fmb,torch::Tensor nw,torch::Tensor nb,"
         "torch::Tensor sw,torch::Tensor sb){"
         "l3_fwd(h_out.data_ptr<float>(),h_fast_out.data_ptr<float>(),"
         "sa_a.data_ptr<float>(),sa_b.data_ptr<float>(),sa_xp.data_ptr<float>(),"
         "sa_field.data_ptr<float>(),sa_fm_out.data_ptr<float>(),sa_normed.data_ptr<float>(),sa_gate.data_ptr<float>(),"
         "h.data_ptr<float>(),x.data_ptr<float>(),"
         "gw.data_ptr<float>(),gb.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),"
         "pw.data_ptr<float>(),pb.data_ptr<float>(),P.data_ptr<float>(),"
         "fmw.data_ptr<float>(),fmb.data_ptr<float>(),nw.data_ptr<float>(),nb.data_ptr<float>(),"
         "sw.data_ptr<float>(),sb.data_ptr<float>(),"
         "h.size(0),h.size(1),gw.size(0));}")
    _K3_FWD = load_inline("k3_fwd_train", cpp_sources=B, cuda_sources=C, functions=["fwd"], verbose=False)
    print("K3 forward kernel ready")

def _load_bwd():
    global _K3_BWD
    if _K3_BWD is not None: return
    _setup_env()
    C_BWD = r"""
__device__ void tree_reduce(volatile float* s, int tid){
    for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}}
__global__ void k3_bwd(
    float* g_hc,float* g_la,float* g_lb,float* g_xo,float* g_fo,float* g_ls,
    float* g_nw,float* g_nb,float* g_Po,float* g_ft,
    const float* g_ho,const float* g_hf,const float* h,const float* x,
    const float* a_a,const float* a_b,const float* a_xp,const float* a_hf,
    const float* a_fd,const float* a_fm,const float* a_nm,const float* a_g,
    const float* gw,const float* gb,const float* fw,const float* fb,
    const float* pw,const float* pb,const float* P,
    const float* fmw,const float* fmb,const float* nw,const float* nb,
    const float* sw,const float* sb,
    int bs,int dm,int ne){
    int e=blockIdx.x,b=blockIdx.y,tid=threadIdx.x;
    if(e>=ne||b>=bs)return;
    extern __shared__ volatile float s_red[];
    int base=e*bs*dm+b*dm;int eb=e*dm,ed=e*dm*dm;float gate=a_g[e*bs+b];
    float gs=0;for(int d=tid;d<dm;d+=blockDim.x)gs+=g_ho[base+d]*a_nm[base+d]*0.1f;
    s_red[tid]=gs;__syncthreads();tree_reduce(s_red,tid);
    float gls=s_red[0]*gate*(1.f-gate);if(tid==0)g_ls[e*bs+b]=gls;
    float mp=0;for(int d=tid;d<dm;d+=blockDim.x)mp+=a_fm[base+d];
    s_red[tid]=mp;__syncthreads();tree_reduce(s_red,tid);float mu=s_red[0]/dm;
    float vp=0;for(int d=tid;d<dm;d+=blockDim.x){float dv=a_fm[base+d]-mu;vp+=dv*dv;}
    s_red[tid]=vp;__syncthreads();tree_reduce(s_red,tid);float rstd=rsqrtf(s_red[0]/dm+1e-5f);
    float sp=0,snp=0;
    for(int d=tid;d<dm;d+=blockDim.x){
        float gn=g_ho[base+d]*gate*0.1f,nv=(a_fm[base+d]-mu)*rstd;
        sp+=gn*nw[eb+d];snp+=gn*nw[eb+d]*nv;}
    s_red[tid]=sp;__syncthreads();tree_reduce(s_red,tid);float sdy=s_red[0];
    s_red[tid]=snp;__syncthreads();tree_reduce(s_red,tid);float sdn=s_red[0];
    for(int d=tid;d<dm;d+=blockDim.x){
        float gn=g_ho[base+d]*gate*0.1f,nv=(a_fm[base+d]-mu)*rstd;
        g_fo[base+d]=gn*nw[eb+d]*rstd-rstd/dm*sdy-nv*rstd/dm*sdn;
        g_nw[eb+d]+=gn*nv;g_nb[eb+d]+=gn;}
    __syncthreads();
    for(int d=tid;d<dm;d+=blockDim.x){
        float gf=0;for(int j=0;j<dm;j++)gf+=g_fo[base+j]*fmw[ed+j*dm+d];g_ft[base+d]=gf;}
    __syncthreads();
    for(int d=tid;d<dm;d+=blockDim.x){
        float ghf=0;for(int j=0;j<dm;j++)ghf+=g_ft[base+j]*P[ed+j*dm+d];
        float gt=g_ho[base+d]+ghf+g_hf[base+d];
        g_hc[base+d]=gt*a_a[base+d];g_xo[base+d]=gt*a_b[base+d];
        g_la[base+d]=gt*h[base+d]*a_a[base+d]*(1.f-a_a[base+d]);
        g_lb[base+d]=gt*a_xp[base+d]*a_b[base+d]*(1.f-a_b[base+d]);
        for(int j=0;j<dm;j++)atomicAdd(&g_Po[ed+d*dm+j],a_hf[base+d]*g_ft[base+j]);}
}
void launch_bwd(float* g_hc,float* g_la,float* g_lb,float* g_xo,float* g_fo,float* g_ls,
    float* g_nw,float* g_nb,float* g_Po,float* g_ft,
    const float* g_ho,const float* g_hf,const float* h,const float* x,
    const float* a_a,const float* a_b,const float* a_xp,const float* a_hf,
    const float* a_fd,const float* a_fm,const float* a_nm,const float* a_g,
    const float* gw,const float* gb,const float* fw,const float* fb,
    const float* pw,const float* pb,const float* P,
    const float* fmw,const float* fmb,const float* nw,const float* nb,
    const float* sw,const float* sb,
    int bs,int dm,int ne){
    dim3 grid(ne, bs);
    k3_bwd<<<grid,256,256*sizeof(float)>>>(g_hc,g_la,g_lb,g_xo,g_fo,g_ls,
        g_nw,g_nb,g_Po,g_ft,
        g_ho,g_hf,h,x,
        a_a,a_b,a_xp,a_hf,
        a_fd,a_fm,a_nm,a_g,
        gw,gb,fw,fb,
        pw,pb,P,
        fmw,fmb,nw,nb,
        sw,sb,bs,dm,ne);}
"""
    B_BWD = ("void launch_bwd(float*,float*,float*,float*,float*,float*,float*,float*,float*,float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "const float*,const float*,const float*,const float*,const float*,const float*,const float*,"
             "const float*,const float*,"
             "int,int,int);"
             "void bwd(torch::Tensor g_hc,torch::Tensor g_la,torch::Tensor g_lb,torch::Tensor g_xo,"
             "torch::Tensor g_fo,torch::Tensor g_ls,"
             "torch::Tensor g_nw,torch::Tensor g_nb,torch::Tensor g_Po,torch::Tensor g_ft,"
             "torch::Tensor g_ho,torch::Tensor g_hf,"
             "torch::Tensor h,torch::Tensor x,"
             "torch::Tensor a_a,torch::Tensor a_b,torch::Tensor a_xp,torch::Tensor a_hf,"
             "torch::Tensor a_fd,torch::Tensor a_fm,torch::Tensor a_nm,torch::Tensor a_g,"
             "torch::Tensor gw,torch::Tensor gb,torch::Tensor fw,torch::Tensor fb,"
             "torch::Tensor pw,torch::Tensor pb,torch::Tensor P,"
             "torch::Tensor fmw,torch::Tensor fmb,torch::Tensor nw,torch::Tensor nb,"
             "torch::Tensor sw,torch::Tensor sb){"
             "launch_bwd(g_hc.data_ptr<float>(),g_la.data_ptr<float>(),g_lb.data_ptr<float>(),g_xo.data_ptr<float>(),"
             "g_fo.data_ptr<float>(),g_ls.data_ptr<float>(),"
             "g_nw.data_ptr<float>(),g_nb.data_ptr<float>(),g_Po.data_ptr<float>(),g_ft.data_ptr<float>(),"
             "g_ho.data_ptr<float>(),g_hf.data_ptr<float>(),"
             "h.data_ptr<float>(),x.data_ptr<float>(),"
             "a_a.data_ptr<float>(),a_b.data_ptr<float>(),a_xp.data_ptr<float>(),a_hf.data_ptr<float>(),"
             "a_fd.data_ptr<float>(),a_fm.data_ptr<float>(),a_nm.data_ptr<float>(),a_g.data_ptr<float>(),"
             "gw.data_ptr<float>(),gb.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),"
             "pw.data_ptr<float>(),pb.data_ptr<float>(),P.data_ptr<float>(),"
             "fmw.data_ptr<float>(),fmb.data_ptr<float>(),nw.data_ptr<float>(),nb.data_ptr<float>(),"
             "sw.data_ptr<float>(),sb.data_ptr<float>(),"
             "h.size(0),h.size(1),gw.size(0));}")
    _K3_BWD = load_inline("k3_bwd_train", cpp_sources=B_BWD, cuda_sources=C_BWD, functions=["bwd"], verbose=False)
    print("K3 backward kernel ready")

def pack_weights(model):
    return (
        torch.stack([e.gate_a.weight for e in model.experts]),
        torch.stack([e.gate_a.bias for e in model.experts]),
        torch.stack([e.gate_b.weight for e in model.experts]),
        torch.stack([e.gate_b.bias for e in model.experts]),
        torch.stack([e.proj_in.weight for e in model.experts]),
        torch.stack([e.proj_in.bias for e in model.experts]),
        torch.stack([e.patterns.T @ e.patterns for e in model.experts]),
        torch.stack([e.field_mix.weight for e in model.experts]),
        torch.stack([e.field_mix.bias for e in model.experts]),
        torch.stack([e.norm.weight for e in model.experts]),
        torch.stack([e.norm.bias for e in model.experts]),
        torch.stack([e.slow_gate.weight.squeeze(0) for e in model.experts]),
        torch.stack([e.slow_gate.bias.squeeze(0) for e in model.experts]),
    )


class FusedExpertFunction(torch.autograd.Function):
    _last_grads = None

    @staticmethod
    def forward(ctx, h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb):
        _load_fwd()
        ne, bs, dm = gw.shape[0], h.shape[0], h.shape[1]
        ctx.ne, ctx.bs, ctx.dm = ne, bs, dm
        h_out = torch.empty(ne, bs, dm, device=h.device)
        h_fast = torch.empty(ne, bs, dm, device=h.device)
        sa = dict(sa_a=torch.empty(ne, bs, dm, device=h.device),
                  sa_b=torch.empty(ne, bs, dm, device=h.device),
                  sa_xp=torch.empty(ne, bs, dm, device=h.device),
                  sa_field=torch.empty(ne, bs, dm, device=h.device),
                  sa_fm_out=torch.empty(ne, bs, dm, device=h.device),
                  sa_normed=torch.empty(ne, bs, dm, device=h.device),
                  sa_gate=torch.empty(ne, bs, device=h.device))
        _K3_FWD.fwd(h_out, h_fast, sa['sa_a'], sa['sa_b'], sa['sa_xp'],
                    sa['sa_field'], sa['sa_fm_out'], sa['sa_normed'], sa['sa_gate'],
                    h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb)
        sa['h_fast'] = h_fast
        ctx.save_for_backward(h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb)
        ctx._saved = sa
        return h_out, h_fast

    @staticmethod
    def backward(ctx, grad_h_out, grad_h_fast):
        _load_bwd()
        h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb = ctx.saved_tensors
        ne, bs, dm = ctx.ne, ctx.bs, ctx.dm
        s = ctx._saved

        g_hc = torch.empty(ne, bs, dm, device=h.device)
        g_la = torch.empty(ne, bs, dm, device=h.device)
        g_lb = torch.empty(ne, bs, dm, device=h.device)
        g_xo = torch.empty(ne, bs, dm, device=h.device)
        g_fo = torch.empty(ne, bs, dm, device=h.device)
        g_ls = torch.empty(ne, bs, device=h.device)
        g_nw = torch.zeros(ne, dm, device=h.device)
        g_nb = torch.zeros(ne, dm, device=h.device)
        g_Po = torch.zeros(ne, dm, dm, device=h.device)
        g_ft = torch.empty(ne, bs, dm, device=h.device)

        _K3_BWD.bwd(g_hc, g_la, g_lb, g_xo, g_fo, g_ls,
                    g_nw, g_nb, g_Po, g_ft,
                    grad_h_out, grad_h_fast, h, x_emb,
                    s['sa_a'], s['sa_b'], s['sa_xp'], s['h_fast'],
                    s['sa_field'], s['sa_fm_out'], s['sa_normed'], s['sa_gate'],
                    gw, gb, fw, fb, pw, pb, P,
                    fmw, fmb, nw, nb, sw, sb)

        combined = torch.cat([h, x_emb], dim=-1)
        cn = combined.unsqueeze(0).expand(ne, -1, -1)
        gca = torch.einsum('nbd,ndk->nbk', g_la, gw)
        gcb = torch.einsum('nbd,ndk->nbk', g_lb, fw)
        gcs = torch.einsum('nb,nk->nbk', g_ls, sw)
        gxp = torch.einsum('nbd,ndk->nbk', g_xo, pw)
        grad_h = (g_hc.sum(0) + gca[:,:,:dm].sum(0) + gcb[:,:,:dm].sum(0) + gcs[:,:,:dm].sum(0))
        grad_x = (gca[:,:,dm:].sum(0) + gcb[:,:,dm:].sum(0) + gcs[:,:,dm:].sum(0) + gxp.sum(0))

        gs = dict(grad_logit_a=g_la.detach(), grad_logit_b=g_lb.detach(),
                  grad_xp_out=g_xo.detach(), grad_fm_out=g_fo.detach(),
                  grad_logit_s=g_ls.detach(), grad_nw=g_nw, grad_nb=g_nb,
                  combined_ne=cn, x_emb=x_emb, sa_field=s['sa_field'])
        if FusedExpertFunction._last_grads is None:
            gs.update(gw=gw, gb=gb, fw=fw, fb=fb, pw=pw, pb=pb, fmw=fmw, fmb=fmb,
                      nw=nw, nb=nb, sw=sw, sb=sb)
            FusedExpertFunction._last_grads = gs
        else:
            for k in ['grad_logit_a', 'grad_logit_b', 'grad_xp_out', 'grad_fm_out',
                      'grad_logit_s', 'grad_nw', 'grad_nb']:
                FusedExpertFunction._last_grads[k] += gs[k]
        return grad_h, grad_x, None, None, None, None, None, None, None, None, None, None, None, None, None


def compute_param_grads():
    s = FusedExpertFunction._last_grads
    if s is None: return None
    ne = s['gw'].shape[0]
    grads = {}
    grads['gate_a.weight'] = torch.einsum('nbd,nbk->ndk', s['grad_logit_a'], s['combined_ne'])
    grads['gate_a.bias'] = s['grad_logit_a'].sum(1)
    grads['gate_b.weight'] = torch.einsum('nbd,nbk->ndk', s['grad_logit_b'], s['combined_ne'])
    grads['gate_b.bias'] = s['grad_logit_b'].sum(1)
    grads['proj_in.weight'] = torch.einsum('nbd,nbk->ndk', s['grad_xp_out'], s['x_emb'].unsqueeze(0).expand(ne, -1, -1))
    grads['proj_in.bias'] = s['grad_xp_out'].sum(1)
    grads['field_mix.weight'] = torch.einsum('nbd,nbk->ndk', s['grad_fm_out'], s['sa_field'])
    grads['field_mix.bias'] = s['grad_fm_out'].sum(1)
    grads['norm.weight'] = s['grad_nw']
    grads['norm.bias'] = s['grad_nb']
    grads['slow_gate.weight'] = torch.einsum('nb,nbk->nk', s['grad_logit_s'], s['combined_ne'])
    grads['slow_gate.bias'] = s['grad_logit_s'].sum(1)
    FusedExpertFunction._last_grads = None
    return grads


def apply_param_grads(model, grads):
    for e, expert in enumerate(model.experts):
        expert.gate_a.weight.grad = grads['gate_a.weight'][e]
        expert.gate_a.bias.grad = grads['gate_a.bias'][e]
        expert.gate_b.weight.grad = grads['gate_b.weight'][e]
        expert.gate_b.bias.grad = grads['gate_b.bias'][e]
        expert.proj_in.weight.grad = grads['proj_in.weight'][e]
        expert.proj_in.bias.grad = grads['proj_in.bias'][e]
        expert.field_mix.weight.grad = grads['field_mix.weight'][e]
        expert.field_mix.bias.grad = grads['field_mix.bias'][e]
        expert.norm.weight.grad = grads['norm.weight'][e]
        expert.norm.bias.grad = grads['norm.bias'][e]
        expert.slow_gate.weight.grad = grads['slow_gate.weight'][e].unsqueeze(0)
        expert.slow_gate.bias.grad = grads['slow_gate.bias'][e].unsqueeze(0)
