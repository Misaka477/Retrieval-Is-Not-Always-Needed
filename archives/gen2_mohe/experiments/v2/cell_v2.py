"""
CANN-SSM Cell V2 — 支持批量化 attractor（adiabatic elimination 快慢分离）。

与 v1 核心区别：
- forward_batch_attractor(h_ssm_stacked, combined_stacked) 一次处理 K steps
- M = batch * K，GPU 利用率随 attract_every 线性增长
- gate 保持不变（per-step dense GEMM，保交叉维度混合）
"""
import torch
import torch.nn as nn


class CANNSSMCellV2(nn.Module):
    """CANN-SSM cell with batched attractor support.

    参数与 v1 完全相同，额外提供 forward_batch_attractor() 用于
    adiabatic elimination 路径。
    """

    def __init__(self, d_model, n_patterns=4096, beta=1.0, attract_every=2,
                 pattern_rank=0, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.rank = pattern_rank
        self.n_heads = n_heads

        np_per_head = n_patterns // n_heads
        if pattern_rank > 0 and pattern_rank < n_patterns:
            self.U = nn.Parameter(torch.randn(n_patterns, pattern_rank) * 0.02)
            self.V = nn.Parameter(torch.randn(pattern_rank, d_model) * 0.02 / (pattern_rank ** 0.5))
            self.patterns = None
        elif n_heads > 1:
            self.patterns = nn.Parameter(torch.randn(n_heads, np_per_head, d_model) * 0.02)
            self.U = None
            self.V = None
        else:
            self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
            self.U = None
            self.V = None

        self.register_buffer("beta_t", torch.tensor([beta]))
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.gate_alpha = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    @property
    def effective_patterns(self):
        if self.patterns is not None:
            return self.patterns
        return self.U @ self.V

    # ── Per-step gate (identical to v1) ──
    def forward_gate(self, h, x):
        """SSM gate only: h_ssm = a*h + b*(x @ Wp + bp), 然后 LayerNorm。"""
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(combined @ self.gate_a.weight.T + self.gate_a.bias)
        b = torch.sigmoid(combined @ self.gate_b.weight.T + self.gate_b.bias)
        xp = x @ self.proj_in.weight.T + self.proj_in.bias
        h_ssm = a * h + b * xp
        combined_out = torch.cat([h, x], dim=-1)
        return h_ssm, combined_out

    # ── Single-step forward (unchanged from v1, used in reference path) ──
    def forward(self, h, x):
        pat_eff = self.effective_patterns
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(combined @ self.gate_a.weight.T + self.gate_a.bias)
        b = torch.sigmoid(combined @ self.gate_b.weight.T + self.gate_b.bias)
        xp = x @ self.proj_in.weight.T + self.proj_in.bias
        h_ssm = a * h + b * xp

        bsz = h.shape[0]
        pat = pat_eff.unsqueeze(0).expand(bsz, -1, -1)
        xi = h_ssm.unsqueeze(1)
        scores = xi @ pat.transpose(1, 2) * self.beta_t[0]
        attn = torch.softmax(scores, dim=-1)
        h_attracted = (attn @ pat).squeeze(1)

        alpha = torch.sigmoid(combined @ self.gate_alpha.weight.T + self.gate_alpha.bias)
        h_new = h_ssm + alpha * (h_attracted - h_ssm)
        return self.norm(h_new)

    # ── Batched attractor (the v2 core) ──
    def forward_batch_attractor(self, h_ssm_stacked, combined_stacked):
        """Batched attractor: 一次处理 K steps。

        Args:
            h_ssm_stacked: [K * batch, d_model]  — stacked SSM outputs
            combined_stacked: [K * batch, 2 * d_model]  — stacked [h, x] 拼接

        Returns:
            h_new: [K * batch, d_model]  — attractor-corrected states
        """
        pat_eff = self.effective_patterns
        M = h_ssm_stacked.shape[0]

        pat = pat_eff.unsqueeze(0).expand(M, -1, -1)
        xi = h_ssm_stacked.unsqueeze(1)
        scores = xi @ pat.transpose(1, 2) * self.beta_t[0]
        attn = torch.softmax(scores, dim=-1)
        h_attracted = (attn @ pat).squeeze(1)

        alpha = torch.sigmoid(
            combined_stacked @ self.gate_alpha.weight.T + self.gate_alpha.bias
        )
        h_new = h_ssm_stacked + alpha * (h_attracted - h_ssm_stacked)
        return h_new
