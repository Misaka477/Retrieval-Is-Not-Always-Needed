"""Fused CUDA kernels for MoHE expert computation."""
import os, glob, subprocess
import torch

# Auto-detect GPU compute capability
_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"

from torch.utils.cpp_extension import load_inline

_K1 = _K2 = None

def _find_msvc():
    """Find MSVC cl.exe via registry or common paths."""
    # 1. Try reading VS install path from registry
    try:
        import winreg
        for key in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
            for sub in [r"SOFTWARE\Microsoft\VisualStudio\Setup\VS",
                        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\Setup\VS"]:
                try:
                    k = winreg.OpenKey(key, sub)
                    for i in range(winreg.QueryInfoKey(k)[0]):
                        try:
                            vs_path, _ = winreg.QueryValueEx(winreg.OpenKey(k, winreg.EnumKey(k, i)), "ProductDir")
                            cl = glob.glob(os.path.join(vs_path, r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
                            if cl:
                                return os.path.dirname(cl[0])
                        except: pass
                except: pass
        # 2. Registry fallback: vswhere
        vsw = glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if vsw:
            r = subprocess.run([vsw[0], "-property", "installationPath", "-latest"],
                             capture_output=True, text=True)
            vs_path = r.stdout.strip()
            cl = glob.glob(os.path.join(vs_path, r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
            if cl:
                return os.path.dirname(cl[0])
    except: pass
    # 3. Common paths (Linux/macOS don't need MSVC)
    for p in [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
        r"D:\Software_Development\Microsoft Visual Studio\2022\Community",
    ]:
        cl = glob.glob(os.path.join(p, r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
        if cl:
            return os.path.dirname(cl[0])
    return None

def _load():
    global _K1, _K2
    if _K1 is not None: return

    cl_dir = _find_msvc()
    if cl_dir:
        os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")
    cu = os.environ.get("CUDA_PATH", r"D:\Software_Development\CUDA_Toolkit_12.4")
    os.environ["PATH"] = os.path.join(cu, "bin") + os.pathsep + os.environ.get("PATH", "")

def _load():
    global _K1, _K2
    if _K1 is not None: return

    # Ensure MSVC is in PATH for compilation
    msvc_dir = r"D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
    cuda_dir = r"D:\Software_Development\CUDA_Toolkit_12.4\bin"
    os.environ["PATH"] = msvc_dir + os.pathsep + cuda_dir + os.pathsep + os.environ.get("PATH", "")

    C1 = r"""
__global__ void k1(float* hf,const float* h,const float* x,
    const float* gw,const float* gb,const float* fw,const float* fb,
    const float* pw,const float* pb,int bs,int dm){
    int b=blockIdx.x,d=threadIdx.x;if(b>=bs||d>=dm)return;
    float sa=gb[d],sb=fb[d],xp=pb[d];
    for(int k=0;k<dm;k++){
        float hk=h[b*dm+k],xk=x[b*dm+k];
        sa+=hk*gw[k*dm+d]+xk*gw[(k+dm)*dm+d];
        sb+=hk*fw[k*dm+d]+xk*fw[(k+dm)*dm+d];
        xp+=xk*pw[k*dm+d];
    }
    hf[b*dm+d]=__frcp_rn(1+__expf(-fminf(fmaxf(sa,-30.f),30.f)))*h[b*dm+d]
              +__frcp_rn(1+__expf(-fminf(fmaxf(sb,-30.f),30.f)))*xp;}
void l1(float*hf,const float*h,const float*x,const float*gw,const float*gb,
    const float*fw,const float*fb,const float*pw,const float*pb,int bs,int dm){
    k1<<<bs,dm>>>(hf,h,x,gw,gb,fw,fb,pw,pb,bs,dm);}
"""
    B1 = ("void l1(float*,const float*,const float*,const float*,const float*,"
          "const float*,const float*,const float*,const float*,int,int);"
          "void f1(torch::Tensor hf,torch::Tensor h,torch::Tensor x,"
          "torch::Tensor gw,torch::Tensor gb,torch::Tensor fw,torch::Tensor fb,"
          "torch::Tensor pw,torch::Tensor pb){"
          "l1(hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
          "gw.data_ptr<float>(),gb.data_ptr<float>(),fw.data_ptr<float>(),"
          "fb.data_ptr<float>(),pw.data_ptr<float>(),pb.data_ptr<float>(),"
          "h.size(0),h.size(1));}")

    C2 = r"""
__global__ void k2(float* ho,const float* hf,const float*h,const float*x,
    const float*P,const float*fw,const float*fb,
    const float*nw,const float*nb,const float*sw,const float*sb,int bs,int dm){
    extern __shared__ float sh[];int b=blockIdx.x,d=threadIdx.x;
    if(b>=bs||d>=dm)return;
    sh[d]=hf[b*dm+d];__syncthreads();
    float fd=0;for(int j=0;j<dm;j++)fd+=sh[j]*P[j*dm+d];
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
    B2 = ("void l2(float*,const float*,const float*,const float*,const float*,"
          "const float*,const float*,const float*,const float*,const float*,const float*,int,int);"
          "void f2(torch::Tensor ho,torch::Tensor hf,torch::Tensor h,torch::Tensor x,"
          "torch::Tensor P,torch::Tensor fw,torch::Tensor fb,"
          "torch::Tensor nw,torch::Tensor nb,torch::Tensor sw,torch::Tensor sb){"
          "l2(ho.data_ptr<float>(),hf.data_ptr<float>(),h.data_ptr<float>(),x.data_ptr<float>(),"
          "P.data_ptr<float>(),fw.data_ptr<float>(),fb.data_ptr<float>(),"
          "nw.data_ptr<float>(),nb.data_ptr<float>(),sw.data_ptr<float>(),sb.data_ptr<float>(),"
          "h.size(0),h.size(1));}")

    _K1 = load_inline("k1", cpp_sources=B1, cuda_sources=C1, functions=["f1"], verbose=False)
    _K2 = load_inline("k2", cpp_sources=B2, cuda_sources=C2, functions=["f2"], verbose=False)

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
