"""RWKV-v7 compatible model. Uses our WKV7 kernel + official untied head + GroupNorm + learnable time_shift."""
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

class WKV7_TimeMix(nn.Module):
    """WKV7 time-mixing with learnable time_shift + output gate + GroupNorm.
    Uses our WKV7 kernel (fixed-decay w = exp(-exp(tmix_w)))."""
    def __init__(self, dm, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        C = dm; H = C // 64; N = 64

        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))

        self.tmix_w = nn.Parameter(torch.randn(H, N) * 0.01)
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.alpha = nn.Linear(C, C, bias=False)
        self.gate = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        self.r_k = nn.Parameter(torch.zeros(H, N))
        self.k_k = nn.Parameter(torch.ones(1, 1, C))

    def forward(self, x, v_first):
        B, T, C = x.shape; H = C // 64; N = 64
        xx = F.pad(x[:, 1:], (0, 0, 0, 1)) - x

        xr = x + xx * self.x_r
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        a = self.alpha(xa)
        g = self.gate(xg)

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)

        w = torch.exp(-torch.exp(self.tmix_w))
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()

        y = WKV7Fn.apply(
            r.view(B, T, H, N).contiguous(),
            w4d,
            k.view(B, T, H, N).contiguous(),
            v.view(B, T, H, N).contiguous(),
            -kk.view(B, T, H, N).contiguous(),
            (kk * torch.sigmoid(a)).view(B, T, H, N).contiguous()
        ).view(B, T, C)

        y = self.ln_x(y.view(B * T, C)).view(B, T, C)
        y = y + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        y = self.output(y * torch.sigmoid(g))
        return y, v_first

class FFN_ChannelMix(nn.Module):
    """RWKV channel-mixing with learnable time_shift + squared ReLU."""
    def __init__(self, dm):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(1, 1, dm))
        self.key = nn.Linear(dm, dm * 4, bias=False)
        self.value = nn.Linear(dm * 4, dm, bias=False)

    def forward(self, x):
        xx = F.pad(x[:, 1:], (0, 0, 0, 1)) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(nn.Module):
    """RWKV Block: LN → TimeMix → +residual → LN → FFN → +residual."""
    def __init__(self, dm, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(dm)
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
        self.att = WKV7_TimeMix(dm, layer_id)
        self.ffn = FFN_ChannelMix(dm)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV7_Model(nn.Module):
    """RWKV-v7 model: untied head, GroupNorm, learnable time_shift, fixed-decay WKV."""
    def __init__(self, vocab_size, dm, n_layers=6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dm)
        self.blocks = nn.ModuleList([Block(dm, i) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x
