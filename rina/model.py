"""MoHE-RWKV v7 CUDA. Pre-hardcoded kernel. Float32."""
import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.cpp_extension import load

CONV_THRESH = 0.05; HEBB_LR = 0.05; INHIBIT_LR = 0.5
_WKVC = None

def _load_wkv7():
    global _WKVC
    if _WKVC is not None: return
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _WKVC = load(name="rwkv7_clampw",
        sources=[os.path.join(root, "kernels/rwkv7_clampw.cu"), os.path.join(root, "kernels/rwkv7_clampw.cpp")],
        extra_cuda_cflags=["-D_N_=64", "-O3"], is_python_module=False, verbose=False)

class WKV7Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B,T,H,N = r.shape
        y = torch.empty(B,T,H,N, device=r.device, dtype=torch.float32)
        s = torch.zeros(B, H, (T+15)//16, N, N, device=r.device, dtype=torch.float32)
        sa = torch.zeros(B, T, H, N, device=r.device, dtype=torch.float32)
        ctx.save_for_backward(r,w,k,v,a,b,s,sa)
        torch.ops.rwkv7_clampw.forward(r.contiguous(),w.contiguous(),k.contiguous(),v.contiguous(),
                                        a.contiguous(),b.contiguous(),y,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        r,w,k,v,a,b,s,sa = ctx.saved_tensors
        dr = torch.empty_like(r); dw = torch.empty_like(r); dk = torch.empty_like(r)
        dv = torch.empty_like(r); da = torch.empty_like(r); db = torch.empty_like(r)
        torch.ops.rwkv7_clampw.backward(r.contiguous(),w.contiguous(),k.contiguous(),v.contiguous(),
                                         a.contiguous(),b.contiguous(),dy.contiguous(),
                                         s,sa,dr,dw,dk,dv,da,db)
        return dr,dw,dk,dv,da,db

class AttractorExpert(nn.Module):
    def __init__(self, dm, np_, name=""):
        super().__init__()
        self.name = name; self.proj = nn.Sequential(nn.Linear(dm,dm*2,bias=False),nn.GELU(),nn.Linear(dm*2,dm,bias=False))
        self.slow_gate = nn.Linear(dm*2,1); self.field_mix = nn.Linear(dm,dm); self.norm = nn.LayerNorm(dm)
        self.patterns = nn.Parameter(torch.randn(np_,dm)*0.02)
    def forward(self, h_in, x_emb):
        P = self.patterns.T @ self.patterns; field = h_in @ P; field = self.proj(field)
        field = self.field_mix(field); field = self.norm(field)
        gate = torch.sigmoid(self.slow_gate(torch.cat([h_in, x_emb], dim=-1)))
        return h_in + gate * field

class MoHERWKV(nn.Module):
    def __init__(self, vocab, dm, np_, n_experts=4, aux_loss_weight=0.1, route_noise=0.0, topk=0):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight; self.route_noise = route_noise; self.dm = dm; self.n_experts = n_experts
        _load_wkv7()
        self.embed = nn.Embedding(vocab,dm); self.embed.weight.data.normal_(0,0.05); self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm,vocab,bias=True); self.head.weight = self.embed.weight; self.head.bias.data.fill_(-10.8)
        self.router = nn.Linear(dm,n_experts)
        self.experts = nn.ModuleList([AttractorExpert(dm,np_,name=f"exp_{i}") for i in range(n_experts)])
        self.tmix_w = nn.Parameter(torch.randn(dm // 64, 64) * 0.01); self.tmix_r = nn.Linear(dm,dm,bias=False)
        self.tmix_k = nn.Linear(dm,dm,bias=False); self.tmix_v = nn.Linear(dm,dm,bias=False)
        self.tmix_a = nn.Linear(dm,dm,bias=False)
        self.consolidate = nn.Linear(dm*n_experts,dm); self.consolidate_norm = nn.LayerNorm(dm)
        self.topk = topk
        self.router_bias = nn.Parameter(torch.randn(n_experts) * 0.5); self.expert_norm = nn.LayerNorm(dm)
        self.router.weight.data.mul_(2.0)

    def forward(self, x):
        bsz, seq_len = x.shape; device = x.device;         emb = self.embed_norm(self.embed(x))
        B,T,D = emb.shape; H = D // 64; N = 64
        w = torch.exp(-torch.exp(self.tmix_w))  # [H,N]
        r = self.tmix_r(emb); k = self.tmix_k(emb); v = self.tmix_v(emb); a = self.tmix_a(emb) * 0.01
        r4d = r.view(B,T,H,N).contiguous(); k4d = k.view(B,T,H,N).contiguous()
        v4d = v.view(B,T,H,N).contiguous(); a4d = a.view(B,T,H,N).contiguous()
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B,T,H,N).contiguous()
        b4d = a4d.clone()
        h = WKV7Fn.apply(r4d,w4d,k4d,v4d,-a4d,b4d).view(B,T,D)
        
        route_logits = []
        for depth in range(3):
            route_raw = (self.router(h) + self.router_bias) * 3.0
            if self.training and self.route_noise > 0: route_raw += torch.randn_like(route_raw)*self.route_noise
            route_logits.append(route_raw)
            route_weights = torch.softmax(route_raw, dim=-1)
            self._last_route_entropy = -(route_weights*torch.log(route_weights.clamp(min=1e-10))).sum(-1).mean().item()
            h_exps = torch.stack([e(h, emb) for e in self.experts], dim=0)
            h_exps = self.expert_norm(h_exps.permute(1,2,0,3).reshape(-1,D)).reshape(B,T,self.n_experts,D)
            if self.topk > 0 and self.topk < self.n_experts:
                _, inds = route_weights.topk(self.topk,dim=-1)
                mask = torch.zeros(B,T,self.n_experts,device=device).scatter_(-1,inds,1)
                h_exps = h_exps * mask.unsqueeze(-1)
            h_new = self.consolidate_norm(self.consolidate(h_exps.reshape(B,T,self.n_experts*D)))
            h = h_new
        logits = self.head(h_new)
        if self.training and route_logits:
            lf = torch.stack(route_logits,dim=0).transpose(0,1).reshape(-1,self.n_experts)
            p = torch.softmax(lf,dim=-1); f = p.mean(dim=0)
            al = self.aux_loss_weight*self.n_experts*(f*p.mean(0)).sum()
            zl = (lf.logsumexp(-1)**2).mean()*1e-4; rl = 1e-3*(lf**2).mean()
            eb = -0.01 * (-(p * torch.log(p.clamp(min=1e-10))).sum(-1).mean())
            self._last_aux_loss = al+zl+rl+eb
        else: self._last_aux_loss = 0.0
        return logits
    def finish_training_step(self): pass
