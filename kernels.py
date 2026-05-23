"""Fused CUDA kernels for MoHE."""
import os, glob
import torch
_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
from torch.utils.cpp_extension import load_inline

_K1 = _K2 = _K3 = None

def _find_msvc():
    for root in [r"C:\Program Files\Microsoft Visual Studio",
                 r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"D:\Software_Development\Microsoft Visual Studio"]:
        hits = glob.glob(os.path.join(root, r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
        if hits:
            return os.path.dirname(hits[0])
    return None

def _load():
    global _K1, _K2, _K3
    if _K1 is not None: return
    _msvc = _find_msvc()
    if _msvc:
        os.environ["PATH"] = _msvc + os.pathsep + os.environ.get("PATH", "")
    _cu = os.environ.get("CUDA_PATH", r"D:\Software_Development\CUDA_Toolkit_12.4")
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Kernel 1: fast_ssm
    C1 = r"""
__global__ void k1(float* hf,const float*h,const float*x,
    const float*gw,const float*gb,const float*fw,const float*fb,
    const float*pw,const float*pb,int bs,int dm){
    int b=blockIdx.x,d=threadIdx.x;if(b>=bs||d>=dm)return;
    float sa=gb[d],sb=fb[d],xp=pb[d];
    for(int k=0;k<dm;k++){
        float hk=h[b*dm+k],xk=x[b*dm+k];
        sa+=hk*gw[k*dm+d]+xk*gw[(k+dm)*dm+d];
        sb+=hk*fw[k*dm+d]+xk*fw[(k+dm)*dm+d];
        xp+=xk*pw[k*dm+d];}
    hf[b*dm+d]=__frcp_rn(1+__expf(-fminf(fmaxf(sa,-30.f),30.f)))*h[b*dm+d]
              +__frcp_rn(1+__expf(-fminf(fmaxf(sb,-30.f),30.f)))*xp;}
void l1(float*hf,const float*h,const float*x,const float*gw,const float*gb,
    const float*fw,const float*fb,const float*pw,const float*pb,int bs,int dm){
    k1<<<bs,dm>>>(hf,h,x,gw,gb,fw,fb,pw,pb,bs,dm);}
"""
    B1 = "void l1(float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,int,int);"
    B1 += "void f1(torch::Tensor hf,torch::Tensor h,torch::Tensor x,torch::Tensor gw,torch::Tensor gb,torch::Tensor fw,torch::Tensor fb,torch::Tensor pw,torch::Tensor pb){"
    B1 += "l1(hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),gw.data_ptr<float>(),gb.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),pw.data_ptr<float>(),pb.data_ptr<float>(),h.size(0),h.size(1));}"

    # Kernel 2: field + norm + slow_gate
    C2 = r"""
__global__ void k2(float* ho,const float*hf,const float*h,const float*x,
    const float*P,const float*fw,const float*fb,
    const float*nw,const float*nb,const float*sw,const float*sb,int bs,int dm){
    extern __shared__ float sh[];int b=blockIdx.x,d=threadIdx.x;
    if(b>=bs||d>=dm)return;
    sh[d]=hf[b*dm+d];__syncthreads();float fd=0;
    for(int j=0;j<dm;j++)fd+=sh[j]*P[j*dm+d];
    float fm=fb[d];for(int j=0;j<dm;j++){float fj=0;
        for(int k=0;k<dm;k++)fj+=sh[k]*P[k*dm+j];fm+=fj*fw[j*dm+d];}
    sh[d]=fm;__syncthreads();float mn=0,vr=0;
    for(int j=0;j<dm;j++)mn+=sh[j];mn/=dm;
    for(int j=0;j<dm;j++){float dv=sh[j]-mn;vr+=dv*dv;}vr/=dm;
    float fn=(fm-mn)*__frsqrt_rn(vr+1e-5f)*nw[d]+nb[d];
    float sg=sb[0];for(int k=0;k<dm;k++)sg+=h[b*dm+k]*sw[k]+x[b*dm+k]*sw[k+dm];
    float gv=__frcp_rn(1+__expf(-fminf(fmaxf(sg,-30.f),30.f)));
    ho[b*dm+d]=hf[b*dm+d]+gv*fn*0.1f;}
void l2(float*ho,const float*hf,const float*h,const float*x,const float*P,
    const float*fw,const float*fb,const float*nw,const float*nb,
    const float*sw,const float*sb,int bs,int dm){
    k2<<<bs,dm,dm*2*sizeof(float)>>>(ho,hf,h,x,P,fw,fb,nw,nb,sw,sb,bs,dm);}
"""
    B2 = "void l2(float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,int,int);"
    B2 += "void f2(torch::Tensor ho,torch::Tensor hf,torch::Tensor h,torch::Tensor x,torch::Tensor P,torch::Tensor fw,torch::Tensor fb,torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
    B2 += "l2(ho.data_ptr<float>(),hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),P.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),h.size(0),h.size(1));}"

    # Kernel 3: Fused N-expert computation (SSM + field + norm + slow_gate → h_out)
    # Grid: (bs) blocks, Block: (dm) threads, Shared: 2*dm*sizeof(float) per expert
    C3 = r"""
__global__ void k3(float* h_out, float* h_fast_out,
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
    float* s_hf=s;           // s[0..dm) = h_fast / norm input per expert
    float* s_field=s+dm;     // s[dm..2*dm) = field per expert
    float h_val=h[b*dm+d];
    for(int e=0;e<ne;e++){
        int eb=e*dm, e2d=e*2*dm*dm, ed=e*dm*dm;
        // 1. gate_a + gate_b + proj_in → h_fast (__fma_rn)
        float sa=gb[eb+d],sb_val=fb[eb+d],xp=pb[eb+d];
        for(int k=0;k<dm;k++){
            float hk=h[b*dm+k],xk=x[b*dm+k];
            sa=__fma_rn(hk,gw[e2d+d*2*dm+k],sa);
            sa=__fma_rn(xk,gw[e2d+d*2*dm+dm+k],sa);
            sb_val=__fma_rn(hk,fw[e2d+d*2*dm+k],sb_val);
            sb_val=__fma_rn(xk,fw[e2d+d*2*dm+dm+k],sb_val);
            xp=__fma_rn(xk,pw[ed+d*dm+k],xp);}
        float sig_a=__frcp_rn(1.f+expf(-fminf(fmaxf(sa,-30.f),30.f)));
        float sig_b=__frcp_rn(1.f+expf(-fminf(fmaxf(sb_val,-30.f),30.f)));
        float hf=__fma_rn(sig_b,xp,sig_a*h_val);
        s_hf[d]=hf;__syncthreads();
        // 2. field = h_fast @ P (__fma_rn)
        float fd=0;
        for(int j=0;j<dm;j++)fd=__fma_rn(s_hf[j],P[ed+j*dm+d],fd);
        s_field[d]=fd;__syncthreads();
        // 3. field_mix: field @ fmw.T + fmb
        float fm_val=fmb[eb+d];
        for(int j=0;j<dm;j++)fm_val=__fma_rn(s_field[j],fmw[ed+d*dm+j],fm_val);
        s_hf[d]=fm_val;__syncthreads();
        // 4. layer_norm
        float mn=0;for(int j=0;j<dm;j++)mn+=s_hf[j];mn/=dm;
        float vr=0;for(int j=0;j<dm;j++){float dv=s_hf[j]-mn;vr+=dv*dv;}vr/=dm;
        float fn=(fm_val-mn)*__frsqrt_rn(vr+1e-5f)*nw[eb+d]+nb[eb+d];
        // 5. slow_gate (scalar per expert, thread 0 computes)
        float sg_val=sb[e];
        if(d==0){for(int k=0;k<dm;k++){
            sg_val=__fma_rn(h[b*dm+k],sw[e*2*dm+k],sg_val);
            sg_val=__fma_rn(x[b*dm+k],sw[e*2*dm+dm+k],sg_val);}}
        s_field[0]=sg_val;__syncthreads();sg_val=s_field[0];
        float gate=__frcp_rn(1.f+expf(-fminf(fmaxf(sg_val,-30.f),30.f)));
        // 6. h_out = h_fast + gate * normed_field * 0.1
        float h_out_val=__fma_rn(gate*0.1f,fn,hf);
        int base=e*bs*dm+b*dm;
        h_out[base+d]=h_out_val;
        h_fast_out[base+d]=hf;
        __syncthreads();}}
void l3(float* h_out,float* h_fast_out,
    const float* h,const float* x,
    const float* gw,const float* gb,
    const float* fw,const float* fb,
    const float* pw,const float* pb,
    const float* P,
    const float* fmw,const float* fmb,
    const float* nw,const float* nb,
    const float* sw,const float* sb,
    int bs,int dm,int ne){
    k3<<<bs,dm,2*dm*sizeof(float)>>>(h_out,h_fast_out,h,x,
        gw,gb,fw,fb,pw,pb,P,fmw,fmb,nw,nb,sw,sb,bs,dm,ne);}
"""
    B3 = "void l3(float*,float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,int,int,int);"
    B3 += "void f3(torch::Tensor h_out,torch::Tensor h_fast_out,"
    B3 += "torch::Tensor h,torch::Tensor x,"
    B3 += "torch::Tensor gw,torch::Tensor gb,torch::Tensor fw,torch::Tensor fb,torch::Tensor pw,torch::Tensor pb,torch::Tensor P,"
    B3 += "torch::Tensor fmw,torch::Tensor fmb,torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
    B3 += "l3(h_out.data_ptr<float>(),h_fast_out.data_ptr<float>(),"
    B3 += "h.data_ptr<float>(),x.data_ptr<float>(),"
    B3 += "gw.data_ptr<float>(),gb.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),pw.data_ptr<float>(),pb.data_ptr<float>(),P.data_ptr<float>(),"
    B3 += "fmw.data_ptr<float>(),fmb.data_ptr<float>(),nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
    B3 += "h.size(0),h.size(1),gw.size(0));}"

    _K1 = load_inline("k1", cpp_sources=B1, cuda_sources=C1, functions=["f1"], verbose=False)
    _K2 = load_inline("k2", cpp_sources=B2, cuda_sources=C2, functions=["f2"], verbose=False)
    _K3 = load_inline("k3", cpp_sources=B3, cuda_sources=C3, functions=["f3"], verbose=False)
    print(f"Kernels ready (k1 ok, k2 ok, k3 ok)")

def fused_expert(h, x_emb, expert, P):
    _load()
    hf = torch.empty_like(h); ho = torch.empty_like(h)
    _K1.f1(hf, h, x_emb, expert.gate_a.weight, expert.gate_a.bias,
           expert.gate_b.weight, expert.gate_b.bias,
           expert.proj_in.weight, expert.proj_in.bias)
    _K2.f2(ho, hf, h, x_emb, P, expert.field_mix.weight, expert.field_mix.bias,
           expert.norm.weight, expert.norm.bias,
           expert.slow_gate.weight, expert.slow_gate.bias)
    return ho, hf

def fused_all_experts(h, x_emb, model):
    """Fused N-expert computation: SSM + field + norm + slow_gate → h_out.
    
    Returns (h_out_packed, h_fast_packed) each shaped [ne, bs, dm].
    """
    _load()
    ne = len(model.experts)
    bs, dm = h.shape
    h_out_packed = torch.empty(ne, bs, dm, device=h.device)
    h_fast_packed = torch.empty(ne, bs, dm, device=h.device)
    # Pack weights
    gw = torch.stack([e.gate_a.weight for e in model.experts])
    gb = torch.stack([e.gate_a.bias for e in model.experts])
    fw = torch.stack([e.gate_b.weight for e in model.experts])
    fb = torch.stack([e.gate_b.bias for e in model.experts])
    pw = torch.stack([e.proj_in.weight for e in model.experts])
    pb = torch.stack([e.proj_in.bias for e in model.experts])
    P = torch.stack([e.patterns.T @ e.patterns for e in model.experts])
    fmw = torch.stack([e.field_mix.weight for e in model.experts])
    fmb = torch.stack([e.field_mix.bias for e in model.experts])
    nw = torch.stack([e.norm.weight for e in model.experts])
    nb = torch.stack([e.norm.bias for e in model.experts])
    sw = torch.stack([e.slow_gate.weight.squeeze(0) for e in model.experts])
    sb = torch.stack([e.slow_gate.bias.squeeze(0) for e in model.experts])
    _K3.f3(h_out_packed, h_fast_packed, h, x_emb,
           gw, gb, fw, fb, pw, pb, P,
           fmw, fmb, nw, nb, sw, sb)
    return h_out_packed, h_fast_packed
