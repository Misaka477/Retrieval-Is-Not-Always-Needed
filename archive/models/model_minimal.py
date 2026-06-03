"""Minimal single-model: fixed-decay WKV + SwiGLU FFN + time_shift + next-token CE."""
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

class MinimalModel(nn.Module):
    """Fixed-decay WKV + SwiGLU FFN with time_shift. Next-token CE (shifted targets)."""
    def __init__(self, vocab, dm):
        super().__init__()
        self.dm = dm
        _load_wkv7()
        self.embed = nn.Embedding(vocab, dm); self.embed.weight.data.normal_(0, 0.05)
        self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=True)
        self.head.weight = self.embed.weight; self.head.bias.data.fill_(-10.8)

        # Fixed-decay WKV
        self.tmix_w = nn.Parameter(torch.randn(dm // 64, 64) * 0.01)
        self.tmix_r = nn.Linear(dm, dm, bias=False)
        self.tmix_k = nn.Linear(dm, dm, bias=False)
        self.tmix_v = nn.Linear(dm, dm, bias=False)
        self.tmix_a = nn.Linear(dm, dm, bias=False)

        # SwiGLU FFN with time_shift
        self.ffn_gate = nn.Linear(dm, dm * 4, bias=False)
        self.ffn_up = nn.Linear(dm, dm * 4, bias=False)
        self.ffn_down = nn.Linear(dm * 4, dm, bias=False)

    def forward(self, x, wkv_state=None):
        B, T = x.shape; device = x.device
        D = self.dm; H = D // 64; N = 64

        emb = self.embed_norm(self.embed(x))

        # Fixed-decay WKV
        w = torch.exp(-torch.exp(self.tmix_w))
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()
        r = self.tmix_r(emb).view(B, T, H, N).contiguous()
        k = self.tmix_k(emb).view(B, T, H, N).contiguous()
        v = self.tmix_v(emb).view(B, T, H, N).contiguous()
        a = (self.tmix_a(emb) * 0.01).view(B, T, H, N).contiguous()
        b4d = a.clone()

        if self.training:
            h = WKV7Fn.apply(r, w4d, k, v, -a, b4d).view(B, T, D)
        else:
            h, new_state = WKV7Fn.stateful_apply(r, w4d, k, v, -a, b4d, wkv_state)
            new_wkv_state = new_state

        # SwiGLU FFN with time_shift (mix prev timestep's h into current)
        h_shifted = torch.cat([h[:, :1] * 0, h[:, :-1]], dim=1)
        h_mixed = h + h_shifted
        h = h + self.ffn_down(F.silu(self.ffn_gate(h_mixed)) * self.ffn_up(h_mixed))

        logits = self.head(h)
        if not self.training:
            return logits, new_wkv_state
        return logits
