"""MoHE-RWKV with Mamba-style selective decay. Same CUDA kernel, data-dependent w."""
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

from .model import WKV7Fn, AttractorExpert

class MoHESSM(nn.Module):
    """MoHE with data-dependent selective decay (Mamba-style)."""
    def __init__(self, vocab, dm, np_, n_experts=4, aux_loss_weight=0.1, route_noise=0.0, topk=0, wkv_no_grad=False, soft_routing=False):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight; self.route_noise = route_noise
        self.dm = dm; self.n_experts = n_experts; self.wkv_no_grad = wkv_no_grad
        _load_wkv7()
        self.embed = nn.Embedding(vocab, dm); self.embed.weight.data.normal_(0, 0.05)
        self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=True)
        self.head.weight = self.embed.weight; self.head.bias.data.fill_(-10.8)
        self.router = nn.Linear(dm, n_experts)
        self.experts = nn.ModuleList([AttractorExpert(dm, np_, name=f"exp_{i}") for i in range(n_experts)])

        # Selective SSM: data-dependent decay replaces fixed tmix_w
        self.dt_linear = nn.Linear(dm, dm)  # projects emb → per-position dt
        self.dt_linear.bias.data.fill_(0.5)  # init dt≈0.97 so w≈0.38 ≈ original
        self.tmix_r = nn.Linear(dm, dm, bias=False)
        self.tmix_k = nn.Linear(dm, dm, bias=False)
        self.tmix_v = nn.Linear(dm, dm, bias=False)
        self.tmix_a = nn.Linear(dm, dm, bias=False)

        self.consolidate = nn.Linear(dm * n_experts, dm)
        self.consolidate_norm = nn.LayerNorm(dm)
        self.topk = topk; self.soft_routing = soft_routing
        self.router_bias = nn.Parameter(torch.randn(n_experts) * 0.5)
        self.expert_norm = nn.LayerNorm(dm)
        self.router.weight.data.mul_(2.0)

    def forward(self, x, wkv_state=None):
        B, T = x.shape; device = x.device
        emb = self.embed_norm(self.embed(x))
        D = self.dm; H = D // 64; N = 64

        # Data-dependent decay (Mamba-style selective SSM)
        dt = F.softplus(self.dt_linear(emb))           # [B, T, D]
        w4d = torch.exp(-dt.view(B, T, H, N)).contiguous()  # [B, T, H, N]

        r = self.tmix_r(emb).view(B, T, H, N).contiguous()
        k = self.tmix_k(emb).view(B, T, H, N).contiguous()
        v = self.tmix_v(emb).view(B, T, H, N).contiguous()
        a = (self.tmix_a(emb) * 0.01).view(B, T, H, N).contiguous()
        b4d = a.clone()

        new_wkv_state = None
        if self.training and not self.wkv_no_grad:
            h = WKV7Fn.apply(r, w4d, k, v, -a, b4d).view(B, T, D)
        else:
            h, new_wkv_state = WKV7Fn.stateful_apply(r, w4d, k, v, -a, b4d, wkv_state)

        route_logits = []
        for depth in range(3):
            route_raw = (self.router(h) + self.router_bias) * 3.0
            if self.training and self.route_noise > 0:
                route_raw += torch.randn_like(route_raw) * self.route_noise
            route_logits.append(route_raw)
            route_weights = torch.softmax(route_raw, dim=-1)
            self._last_route_entropy = -(route_weights * torch.log(route_weights.clamp(min=1e-10))).sum(-1).mean().item()
            h_exps = torch.stack([e(h, emb) for e in self.experts], dim=0)
            h_exps = self.expert_norm(h_exps.permute(1, 2, 0, 3).reshape(-1, D))
            h_exps = h_exps.reshape(B, T, self.n_experts, D)
            if self.soft_routing:
                h_exps = h_exps * route_weights.unsqueeze(-1)
            elif self.topk > 0 and self.topk < self.n_experts:
                _, inds = route_weights.topk(self.topk, dim=-1)
                mask = torch.zeros(B, T, self.n_experts, device=device).scatter_(-1, inds, 1)
                h_exps = h_exps * mask.unsqueeze(-1)
            h_new = self.consolidate_norm(self.consolidate(h_exps.reshape(B, T, self.n_experts * D)))
            h = h_new

        logits = self.head(h_new)
        if self.training and route_logits:
            lf = torch.stack(route_logits, dim=0).transpose(0, 1).reshape(-1, self.n_experts)
            p = torch.softmax(lf, dim=-1); f = p.mean(dim=0)
            al = self.aux_loss_weight * self.n_experts * (f * p.mean(0)).sum()
            zl = (lf.logsumexp(-1) ** 2).mean() * 1e-4
            rl = 1e-3 * (lf ** 2).mean()
            eb = -0.01 * (-(p * torch.log(p.clamp(min=1e-10))).sum(-1).mean())
            self._last_aux_loss = al + zl + rl + eb
        else:
            self._last_aux_loss = 0.0
        return (logits, new_wkv_state) if not self.training or self.wkv_no_grad else logits
