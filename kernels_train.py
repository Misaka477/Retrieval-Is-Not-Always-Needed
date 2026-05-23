"""K3 training support: fused forward + Python backward + autograd Function."""
import os, glob, torch
from torch.utils.cpp_extension import load_inline

_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
_K3_FWD = None

def _find_msvc():
    for root in [r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"D:\Software_Development\Microsoft Visual Studio"]:
        hits = glob.glob(os.path.join(root, r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
        if hits: return os.path.dirname(hits[0])
    return None

def _load_fwd():
    global _K3_FWD
    if _K3_FWD is not None: return
    _msvc = _find_msvc()
    if _msvc:
        os.environ["PATH"] = _msvc + os.pathsep + os.environ.get("PATH", "")
    _cu = os.environ.get("CUDA_PATH", r"D:\Software_Development\CUDA_Toolkit_12.4")
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")

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
    int b=blockIdx.x,d=threadIdx.x;
    if(b>=bs||d>=dm)return;
    extern __shared__ float s[];
    float* s_hf=s; float* s_fld=s+dm;
    float h_val=h[b*dm+d];
    for(int e=0;e<ne;e++){
        int eb=e*dm, e2d=e*2*dm*dm, ed=e*dm*dm;
        float sa=gb[eb+d],sbv=fb[eb+d],xp=pb[eb+d];
        for(int k=0;k<dm;k++){
            float hk=h[b*dm+k],xk=x[b*dm+k];
            sa=__fma_rn(hk,gw[e2d+d*2*dm+k],sa);
            sa=__fma_rn(xk,gw[e2d+d*2*dm+dm+k],sa);
            sbv=__fma_rn(hk,fw[e2d+d*2*dm+k],sbv);
            sbv=__fma_rn(xk,fw[e2d+d*2*dm+dm+k],sbv);
            xp=__fma_rn(xk,pw[ed+d*dm+k],xp);}
        float sig_a=__frcp_rn(1.f+expf(-fminf(fmaxf(sa,-30.f),30.f)));
        float sig_b=__frcp_rn(1.f+expf(-fminf(fmaxf(sbv,-30.f),30.f)));
        int ebo=e*bs*dm+b*dm;
        sa_a[ebo+d]=sig_a; sa_b[ebo+d]=sig_b; sa_xp[ebo+d]=xp;
        float hf=__fma_rn(sig_b,xp,sig_a*h_val);
        s_hf[d]=hf; __syncthreads();
        float fd=0;
        for(int j=0;j<dm;j++)fd=__fma_rn(s_hf[j],P[ed+j*dm+d],fd);
        s_fld[d]=fd; sa_field[ebo+d]=fd; __syncthreads();
        float fm_val=fmb[eb+d];
        for(int j=0;j<dm;j++)fm_val=__fma_rn(s_fld[j],fmw[ed+d*dm+j],fm_val);
        sa_fm_out[ebo+d]=fm_val;
        s_hf[d]=fm_val; __syncthreads();
        float mn=0; for(int j=0;j<dm;j++)mn+=s_hf[j]; mn/=dm;
        float vr=0; for(int j=0;j<dm;j++){float dv=s_hf[j]-mn;vr+=dv*dv;}vr/=dm;
        float fn=(fm_val-mn)*__frsqrt_rn(vr+1e-5f)*nw[eb+d]+nb[eb+d];
        sa_normed[ebo+d]=fn;
        float sg_val=sb[e];
        if(d==0){for(int k=0;k<dm;k++){
            sg_val=__fma_rn(h[b*dm+k],sw[e*2*dm+k],sg_val);
            sg_val=__fma_rn(x[b*dm+k],sw[e*2*dm+dm+k],sg_val);}}
        s_fld[0]=sg_val; __syncthreads(); sg_val=s_fld[0];
        float gate=__frcp_rn(1.f+expf(-fminf(fmaxf(sg_val,-30.f),30.f)));
        if(d==0) sa_gate[e*bs+b]=gate;
        float h_out_val=__fma_rn(gate*0.1f,fn,hf);
        h_out[ebo+d]=h_out_val; h_fast_out[ebo+d]=hf;
        __syncthreads();}}
void l3_fwd(float* h_out,float* h_fast_out,
    float* sa_a,float* sa_b,float* sa_xp,
    float* sa_field,float* sa_fm_out,float* sa_normed,float* sa_gate,
    const float* h,const float* x,
    const float* gw,const float* gb,
    const float* fw,const float* fb,
    const float* pw,const float* pb,
    const float* P,
    const float* fmw,const float* fmb,
    const float* nw,const float* nb,
    const float* sw,const float* sb,
    int bs,int dm,int ne){
    k3_fwd<<<bs,dm,2*dm*sizeof(float)>>>(h_out,h_fast_out,
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
    """Fused forward + Python backward for N-expert computation."""
    _last_grads = None

    @staticmethod
    def forward(ctx, h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb):
        _load_fwd()
        ne, bs, dm = gw.shape[0], h.shape[0], h.shape[1]
        ctx.ne = ne; ctx.bs = bs; ctx.dm = dm
        h_out = torch.empty(ne, bs, dm, device=h.device)
        h_fast = torch.empty(ne, bs, dm, device=h.device)
        sa = dict(
            sa_a=torch.empty(ne, bs, dm, device=h.device),
            sa_b=torch.empty(ne, bs, dm, device=h.device),
            sa_xp=torch.empty(ne, bs, dm, device=h.device),
            sa_field=torch.empty(ne, bs, dm, device=h.device),
            sa_fm_out=torch.empty(ne, bs, dm, device=h.device),
            sa_normed=torch.empty(ne, bs, dm, device=h.device),
            sa_gate=torch.empty(ne, bs, device=h.device),
        )
        _K3_FWD.fwd(h_out, h_fast, sa['sa_a'], sa['sa_b'], sa['sa_xp'],
                    sa['sa_field'], sa['sa_fm_out'], sa['sa_normed'], sa['sa_gate'],
                    h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb)
        ctx.save_for_backward(h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb)
        ctx._saved = sa
        return h_out, h_fast

    @staticmethod
    def backward(ctx, grad_h_out, grad_h_fast):
        h, x_emb, gw, gb, fw, fb, pw, pb, P, fmw, fmb, nw, nb, sw, sb = ctx.saved_tensors
        ne, bs, dm = ctx.ne, ctx.bs, ctx.dm
        s = ctx._saved
        combined = torch.cat([h, x_emb], dim=-1)
        a = s['sa_a']; b_sig = s['sa_b']; xp = s['sa_xp']
        gate = s['sa_gate']

        gho = grad_h_out; ghf = grad_h_fast
        g_nm = gho * gate.unsqueeze(-1) * 0.1
        fm = s['sa_fm_out']
        mu = fm.mean(dim=-1, keepdim=True)
        var = fm.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
        rstd = var.rsqrt()
        nv = (fm - mu) * rstd
        dy = g_nm * nw.unsqueeze(1)
        dx = (dy - dy.mean(-1, True) - nv * (dy * nv).mean(-1, True)) * rstd
        grad_fm_out = dx
        grad_field = torch.einsum('nbd,ndk->nbk', grad_fm_out, fmw)
        ghf_field = torch.einsum('nbd,nkd->nbk', grad_field, P)
        ghf_total = gho + ghf_field + ghf
        grad_h_contrib = ghf_total * a
        grad_xp_out = ghf_total * b_sig
        grad_logit_a = ghf_total * h.unsqueeze(0) * a * (1 - a)
        grad_logit_b = ghf_total * xp * b_sig * (1 - b_sig)
        combined_ne = combined.unsqueeze(0).expand(ne, -1, -1)
        gca = torch.einsum('nbd,ndk->nbk', grad_logit_a, gw)
        gcb = torch.einsum('nbd,ndk->nbk', grad_logit_b, fw)
        grad_gate = (gho * s['sa_normed'] * 0.1).sum(dim=-1, keepdim=True)
        grad_slow = grad_gate * gate.unsqueeze(-1) * (1 - gate.unsqueeze(-1))
        gcs = grad_slow * sw.unsqueeze(1)
        gxp_wp = torch.einsum('nbd,ndk->nbk', grad_xp_out, pw)
        grad_h = (grad_h_contrib.sum(0) + gca[:,:,:dm].sum(0) + gcb[:,:,:dm].sum(0) + gcs[:,:,:dm].sum(0))
        grad_x = (gca[:,:,dm:].sum(0) + gcb[:,:,dm:].sum(0) + gcs[:,:,dm:].sum(0) + gxp_wp.sum(0))
        # Accumulate activation gradients for param grad computation
        gs = dict(grad_logit_a=grad_logit_a.detach(), grad_logit_b=grad_logit_b.detach(),
                  grad_xp_out=grad_xp_out.detach(), grad_fm_out=grad_fm_out.detach(),
                  grad_logit_s=grad_slow.squeeze(-1).detach(),
                  grad_nw=(g_nm * nv).sum(1), grad_nb=g_nm.sum(1),
                  combined_ne=combined_ne, x_emb=x_emb, sa_field=s['sa_field'])
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
