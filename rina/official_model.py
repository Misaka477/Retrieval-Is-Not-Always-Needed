"""Official RWKV-v7 model, directly from rwkv_v7_demo.py. No custom modifications."""
import os, torch, types, math
import torch.nn as nn, torch.nn.functional as F

HEAD_SIZE = 64
CHUNK_LEN = 16
DTYPE = torch.float32

# ── Official fp32 CUDA kernel (WindBackstepping, forward+backward) ──
from torch.utils.cpp_extension import load
_load_once = load(name='wind_backstepping',
    sources=[os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kernels/wkv7_fp32.cu'),
             os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kernels/wkv7_fp32.cpp')],
    extra_cuda_cflags=[f'-D_C_={HEAD_SIZE}', f'-D_CHUNK_LEN_={CHUNK_LEN}', '--use_fast_math', '-O3'],
    is_python_module=False, verbose=False)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B,T,H,C = w.shape; assert T%CHUNK_LEN==0
        assert all(i.dtype==torch.float32 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C,dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C,dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b,y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b,dy,s,sa,dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def rwkv7_op(r, w, k, v, a, b):
    """CUDA kernel wrapper. Matches rwkv_v7_demo.py's RWKV7_OP signature."""
    B,T,C = r.shape; H = C//HEAD_SIZE; N = HEAD_SIZE
    r4 = r.view(B,T,H,N).contiguous()
    w4 = w.view(B,T,H,N).contiguous()
    k4 = k.view(B,T,H,N).contiguous()
    v4 = v.view(B,T,H,N).contiguous()
    a4 = a.view(B,T,H,N).contiguous()
    b4 = b.view(B,T,H,N).contiguous()
    # Kernel: (w, q, k, v, z, a) → z=-kk, a=b
    y = WindBackstepping.apply(w4, r4, k4, v4, a4, b4)
    return y.view(B,T,C)

# ── Official layers (unmodified from demo, just no JIT) ──
class TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args; self.layer_id = layer_id
        self.head_size = args.head_size_a; self.n_head = args.dim_att // self.head_size
        H = self.n_head; C = args.n_embd
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']: setattr(self,k,nn.Parameter(torch.empty(1,1,C)))
        self.w0 = nn.Parameter(torch.empty(1,1,C)); self.w1 = nn.Parameter(torch.empty(C,64)); self.w2 = nn.Parameter(torch.empty(64,C))
        self.a0 = nn.Parameter(torch.empty(1,1,C)); self.a1 = nn.Parameter(torch.empty(C,64)); self.a2 = nn.Parameter(torch.empty(64,C))
        self.v0 = nn.Parameter(torch.empty(1,1,C)); self.v1 = nn.Parameter(torch.empty(C,32)); self.v2 = nn.Parameter(torch.empty(32,C))
        self.g1 = nn.Parameter(torch.empty(C,128)); self.g2 = nn.Parameter(torch.empty(128,C))
        self.k_k = nn.Parameter(torch.empty(1,1,C)); self.k_a = nn.Parameter(torch.empty(1,1,C)); self.r_k = nn.Parameter(torch.empty(H,64))
        self.receptance = nn.Linear(C,C,bias=False); self.key = nn.Linear(C,C,bias=False)
        self.value = nn.Linear(C,C,bias=False); self.output = nn.Linear(C,C,bias=False)
        self.ln_x = nn.GroupNorm(H,C,eps=64e-5)
    def forward(self, x, v_first):
        B,T,C = x.shape; H = self.n_head
        xx = F.pad(x[:,1:],(0,0,0,1)) - x
        xr = x + xx*self.x_r; xw = x + xx*self.x_w; xk = x + xx*self.x_k
        xv = x + xx*self.x_v; xa = x + xx*self.x_a; xg = x + xx*self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw@self.w1)@self.w2)) - 0.5
        k = self.key(xk); v = self.value(xv)
        if self.layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv@self.v1)@self.v2)
        a = torch.sigmoid(self.a0 + (xa@self.a1)@self.a2)
        g = torch.sigmoid(xg@self.g1)@self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = rwkv7_op(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B*T,C)).view(B,T,C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(-1,keepdim=True)*v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class ChanMix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1,1,args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
    def forward(self, x):
        xx = F.pad(x[:,1:], (0,0,0,1)) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args; self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(args.n_embd) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(args.n_embd); self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = TimeMix(args, layer_id); self.ffn = ChanMix(args)
    def forward(self, x, v_first):
        if self.layer_id == 0: x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx; x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd; args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    def forward(self, idx, return_h=False):
        x = self.emb(idx); v_first = torch.empty_like(x)
        for block in self.blocks: x, v_first = block(x, v_first)
        x = self.ln_out(x)
        if return_h: return self.head(x), x
        return self.head(x)
