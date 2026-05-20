"""
SNN-CANN Cell — 脉冲门控双流 cell.

架构:
  Stream A (Spike):  全量 gate + attractor → 仅活跃维度
  Stream B (Decay):  指数衰减 → 休眠维度
  
  spike_mask = sigmoid(h @ W_spike) > threshold
  h_new = spike_mask * h_attracted + (1-spike_mask) * (h * decay)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SNNCANNCell(nn.Module):
    """脉冲门控 CANN cell — 稀疏状态更新."""

    def __init__(self, d_model, n_patterns=4096, beta=1.0, attract_every=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every

        # 脉冲门控 (轻量: dm → dm, 而非 2*dm → dm)
        self.spike_proj = nn.Linear(d_model, d_model)
        self.register_buffer("spike_threshold", torch.tensor([0.0]))

        # 休眠衰减
        self.decay = nn.Parameter(torch.zeros(d_model))

        # 标准 gate (和 v1 一样)
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.gate_alpha = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Pattern 记忆
        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.register_buffer("beta_t", torch.tensor([beta]))

        # 统计
        self.register_buffer("spike_count", torch.zeros(1))
        self.register_buffer("total_count", torch.zeros(1))

    def forward(self, h, x, step=0):
        bsz, dm = h.shape

        # Step 1: 预测脉冲 (dm→dm, 轻量)
        spike_logit = self.spike_proj(h)
        spike_prob = torch.sigmoid(spike_logit + self.spike_threshold)

        # Hard mask + straight-through gradient
        spike_hard = (spike_prob > 0.5).float()
        spike_mask = spike_hard + spike_prob - spike_prob.detach()

        # 更新统计
        if self.training:
            self.spike_count += spike_hard.sum().detach()
            self.total_count += spike_hard.numel()

        # Step 2: 标准 gate (全维度)
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        xp = self.proj_in(x)
        h_ssm = a * h + b * xp

        # Step 3: Attractor
        if step % self.attract_every == (self.attract_every - 1):
            pat = self.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h_ssm.unsqueeze(1)
            scores = xi @ pat.transpose(1, 2) * self.beta_t[0]
            attn = torch.softmax(scores, dim=-1)
            h_attracted_global = (attn @ pat).squeeze(1)
        else:
            h_attracted_global = h_ssm

        alpha = torch.sigmoid(self.gate_alpha(combined))
        h_attracted = h_ssm + alpha * (h_attracted_global - h_ssm)

        # Step 4: 休眠衰减
        h_decay = h * torch.sigmoid(self.decay).unsqueeze(0)

        # Step 5: 混合 — 脉冲维度用 attractor 修正，休眠维度衰减
        h_new = spike_mask * h_attracted + (1.0 - spike_mask) * h_decay

        return self.norm(h_new)

    @property
    def spike_rate(self):
        if self.total_count.item() == 0:
            return 0.0
        return (self.spike_count / self.total_count).item()


class SNNSeqModel(nn.Module):
    """SNN-CANN 序列模型."""

    def __init__(self, vocab_size, d_model=256, n_patterns=4096,
                 beta=1.0, n_slots=4096, attract_every=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = SNNCANNCell(d_model, n_patterns=n_patterns,
                                beta=beta, attract_every=attract_every)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))
        self.slot_proj = nn.Linear(d_model, d_model)

    def slot_write(self, key_id, value_id):
        with torch.no_grad():
            ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x):
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits = []

        for t in range(seq_len - 1):
            h = self.cell(h, emb[:, t, :], step=t)
            logits.append(self.head(self.state_norm(h)))

        i_ext = self.slot_table[x[:, -1]]
        h = self.cell(h + i_ext, emb[:, -1, :], step=seq_len - 1)
        logits.append(self.head(self.state_norm(h)))

        return torch.stack(logits, dim=1)

    def get_spike_rate(self):
        return self.cell.spike_rate
