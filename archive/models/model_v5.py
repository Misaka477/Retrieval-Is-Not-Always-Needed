"""MoHE-RWKV V5 — self-supervised attractor with masked reconstruction."""
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from .model import WKV7Fn, _load_wkv7

_load_wkv7()


class AttractorExpertV5(nn.Module):
    """Expert with masked reconstruction loss — prevents gate=0 collapse."""
    def __init__(self, dm, np_, name=""):
        super().__init__()
        self.name = name
        proj_hidden = int(dm * 1.5)
        self.proj = nn.Sequential(
            nn.Linear(dm, proj_hidden, bias=False), nn.GELU(), nn.Linear(proj_hidden, dm, bias=False)
        )
        self.slow_gate = nn.Linear(dm * 2, 1)
        self.field_mix = nn.Linear(dm, dm)
        self.norm = nn.LayerNorm(dm)
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)
        self.mask_rate = 0.3

    def forward(self, h_clean, x_emb):
        """Return (h_recon, recon_loss). h_recon = h_masked + gate·field."""
        B, T, D = h_clean.shape
        # randomly mask 30% of dimensions
        mask = torch.rand(B, T, D, device=h_clean.device) > self.mask_rate
        h_masked = h_clean * mask.float()

        scores = h_masked @ self.patterns.T
        scores = torch.relu(scores)  # sparse activation per pattern
        field = scores @ self.patterns
        field = self.proj(field)
        field = self.field_mix(field)
        field = self.norm(field)
        gate = torch.sigmoid(self.slow_gate(torch.cat([h_masked, x_emb], dim=-1)))
        h_recon = h_masked + gate * field

        # reconstruction loss only on masked dimensions
        err = (h_recon - h_clean) ** 2
        recon_loss = err[~mask].mean() if mask.any() else err.mean()
        return h_recon, recon_loss


class MoHERWKV_V5(nn.Module):
    def __init__(self, vocab, dm, np_, n_experts=4, aux_loss_weight=0.1, topk=2, recon_beta=0.5):
        super().__init__()
        self.dm = dm
        self.n_experts = n_experts
        self.aux_loss_weight = aux_loss_weight
        self.topk = topk
        self.recon_beta = recon_beta
        self.embed = nn.Embedding(vocab, dm)
        self.embed.weight.data.normal_(0, 0.05)
        self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.router = nn.Linear(dm, n_experts)
        self.router_bias = nn.Parameter(torch.randn(n_experts) * 0.5)
        self.experts = nn.ModuleList([AttractorExpertV5(dm, np_) for _ in range(n_experts)])
        self.expert_norm = nn.LayerNorm(dm)
        self.consolidate = nn.Linear(dm * n_experts, dm)
        self.consolidate_norm = nn.LayerNorm(dm)
        self.tmix_w = nn.Parameter(torch.randn(dm // 64, 64) * 0.01)
        self.tmix_r = nn.Linear(dm, dm, bias=False)
        self.tmix_k = nn.Linear(dm, dm, bias=False)
        self.tmix_v = nn.Linear(dm, dm, bias=False)
        self.tmix_a = nn.Linear(dm, dm, bias=False)
        self.router.weight.data.mul_(2.0)
        self.head.bias = nn.Parameter(torch.full([vocab], -10.8))
        self.head.bias.data.fill_(-10.8)

    def forward(self, x):
        device, B, T = x.device, *x.shape
        emb = self.embed_norm(self.embed(x))
        H, N = self.dm // 64, 64

        w = torch.exp(-torch.exp(self.tmix_w))
        r = self.tmix_r(emb).view(B, T, H, N).contiguous()
        k = self.tmix_k(emb).view(B, T, H, N).contiguous()
        v = self.tmix_v(emb).view(B, T, H, N).contiguous()
        a = self.tmix_a(emb).view(B, T, H, N).contiguous() * 0.01
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()
        h = WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B, T, self.dm)

        total_recon = 0.0
        for depth in range(3):
            route_raw = (self.router(h) + self.router_bias) * 3.0
            route_weights = torch.softmax(route_raw, dim=-1)

            h_exps, losses = [], []
            for exp in self.experts:
                h_r, rl = exp(h, emb)
                h_exps.append(h_r)
                losses.append(rl)
            h_exps = torch.stack(h_exps, dim=0)
            h_exps = self.expert_norm(h_exps.permute(1, 2, 0, 3).reshape(-1, self.dm))
            h_exps = h_exps.reshape(B, T, self.n_experts, self.dm)

            if self.topk > 0 and self.topk < self.n_experts:
                _, inds = route_weights.topk(self.topk, dim=-1)
                mask = torch.zeros(B, T, self.n_experts, device=device).scatter_(-1, inds, 1)
                h_exps = h_exps * mask.unsqueeze(-1)

            h_new = self.consolidate_norm(
                self.consolidate(h_exps.reshape(B, T, self.n_experts * self.dm))
            )
            h = h_new
            total_recon = total_recon + sum(losses) / len(losses)

        logits = self.head(h_new)

        # expert-direct aux logits (bypass consolidate, per-expert CE signal)
        h_exps_mixed = h_exps.mean(dim=2)  # [B,T,dm], average of topk experts
        self._aux_logits = h_exps_mixed @ self.head.weight.T + self.head.bias

        self._recon_loss = total_recon / 3
        self._aux_loss = self.aux_loss_weight * self.n_experts * (
            torch.softmax(route_raw, dim=-1).mean(0).pow(2).sum()
        )
        return logits
