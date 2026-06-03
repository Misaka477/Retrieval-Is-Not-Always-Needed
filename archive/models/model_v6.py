"""MoHE-v6: 6× [WKV → MoE(topk=1)] with attractor experts, untied head."""
import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.cpp_extension import load

_WKVC = None
def _load_wkv7():
    global _WKVC
    if _WKVC is not None: return
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _WKVC = load(name="rwkv7_clampw",
        sources=[os.path.join(root, "kernels/rwkv7_clampw.cu"), os.path.join(root, "kernels/rwkv7_clampw.cpp")],
        extra_cuda_cflags=["-D_N_=64", "-O3"], is_python_module=False, verbose=False)

from .model import WKV7Fn

class AttractorExpert(nn.Module):
    def __init__(self, dm, np_):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)
        self.proj = nn.Sequential(nn.Linear(dm, dm * 2, bias=False), nn.GELU(), nn.Linear(dm * 2, dm, bias=False))
        self.gate = nn.Linear(dm, 1)

    def forward(self, h):
        P = self.patterns.T @ self.patterns
        field = h @ P
        field = self.proj(field)
        gate = torch.sigmoid(self.gate(h))
        return gate * field

class Layer6(nn.Module):
    """One layer: WKV → LN → router → expert×2 (topk=1)."""
    def __init__(self, dm, np_, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln_wkv = nn.LayerNorm(dm)
        self.tmix_w = nn.Parameter(torch.randn(dm // 64, 64) * 0.01)
        self.tmix_r = nn.Linear(dm, dm, bias=False)
        self.tmix_k = nn.Linear(dm, dm, bias=False)
        self.tmix_v = nn.Linear(dm, dm, bias=False)
        self.tmix_a = nn.Linear(dm, dm, bias=False)

        self.ln_moe = nn.LayerNorm(dm)
        self.router = nn.Linear(dm, 2)
        self.experts = nn.ModuleList([AttractorExpert(dm, np_) for _ in range(2)])

    def forward(self, h):
        B, T, D = h.shape; H = D // 64; N = 64

        # WKV time-mixing
        w = torch.exp(-torch.exp(self.tmix_w))
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()
        r = self.tmix_r(self.ln_wkv(h)).view(B, T, H, N).contiguous()
        k = self.tmix_k(self.ln_wkv(h)).view(B, T, H, N).contiguous()
        v = self.tmix_v(self.ln_wkv(h)).view(B, T, H, N).contiguous()
        a = (self.tmix_a(self.ln_wkv(h)) * 0.01).view(B, T, H, N).contiguous()
        b4d = a.clone()

        h_res = WKV7Fn.apply(r, w4d, k, v, -a, b4d).view(B, T, D)
        h = h + h_res

        # MoE: topk=1 routing
        h_ln = self.ln_moe(h)
        logits = self.router(h_ln)
        inds = logits.argmax(dim=-1, keepdim=True)
        exp_out = torch.where(inds == 1, self.experts[1](h_ln), self.experts[0](h_ln))
        h = h + exp_out
        return h

class MoHEv6(nn.Module):
    """6× [WKV → MoE(topk=1)]. Untied head. Attractor experts."""
    def __init__(self, vocab, dm, np_, n_layers=6):
        super().__init__()
        _load_wkv7()
        self.embed = nn.Embedding(vocab, dm)
        self.embed.weight.data.normal_(0, 0.05)
        self.ln0 = nn.LayerNorm(dm)
        self.layers = nn.ModuleList([Layer6(dm, np_, i) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=False)

    def forward(self, x):
        h = self.ln0(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_out(h))
