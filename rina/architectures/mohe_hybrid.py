"""
MoHE-Hybrid: Independent h_i + lightweight GQA attention (Jamba-inspired).
Attractor layers handle 75% of compute, attention provides induction head at 25% ratio.
"""
import torch, torch.nn as nn, torch.nn.functional as F

CONV_THRESH = 0.05
HEBB_LR = 0.05
INHIBIT_LR = 0.5


class LightweightAttention(nn.Module):
    """GQA attention, no KV cache, for MoHE hybrid architecture."""

    def __init__(self, dm, n_heads=4, n_kv_heads=2):
        super().__init__()
        assert dm % n_heads == 0
        self.dm = dm
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dm // n_heads
        self.q_proj = nn.Linear(dm, dm)
        self.k_proj = nn.Linear(dm, self.head_dim * n_kv_heads)
        self.v_proj = nn.Linear(dm, self.head_dim * n_kv_heads)
        self.o_proj = nn.Linear(dm, dm)
        self.norm = nn.LayerNorm(dm)

    def forward(self, x):
        B, S, D = x.shape
        x = self.norm(x)
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # GQA: expand kv heads to match q heads
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out)


class HybridExpertCell(nn.Module):
    def __init__(self, dm, np_, name=""):
        super().__init__()
        self.name = name
        self.gate_a = nn.Linear(dm, dm)
        self.gate_b = nn.Linear(dm, dm)
        self.proj_in = nn.Linear(dm, dm)
        self.proj = nn.Sequential(
            nn.Linear(dm, dm * 2, bias=False), nn.GELU(), nn.Linear(dm * 2, dm, bias=False),
        )
        self.slow_gate = nn.Linear(dm * 2, 1)
        self.field_mix = nn.Linear(dm, dm)
        self.norm = nn.LayerNorm(dm)
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)

    def _attract(self, h_fast, h_state, x_emb):
        P = self.patterns.T @ self.patterns
        field = h_fast @ P
        field = self.proj(field)
        field = self.field_mix(field)
        field = self.norm(field)
        gate = torch.sigmoid(self.slow_gate(torch.cat([h_state, x_emb], dim=-1)))
        return h_fast + gate * field * 0.1

    def step(self, h_state, x_emb):
        xp = self.proj_in(x_emb)
        a = torch.sigmoid(self.gate_a(h_state))
        b = torch.sigmoid(self.gate_b(h_state))
        h_fast = a * h_state + b * xp
        return self._attract(h_fast, h_state, x_emb), h_fast


class MoHEHybrid(nn.Module):
    def __init__(self, vocab, dm, np_, n_experts=4,
                 aux_loss_weight=0.1, route_noise=0.0, topk=0):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.route_noise = route_noise
        self.embed = nn.Embedding(vocab, dm)
        self.embed.weight.data.normal_(0, 0.05)
        self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=True)
        self.head.weight = self.embed.weight
        self.head.bias.data.fill_(-10.8)
        self.head_bias = -10.8
        self.state_norm = nn.LayerNorm(dm)
        self.router = nn.Linear(dm, n_experts)
        self.experts = nn.ModuleList([
            HybridExpertCell(dm, np_, name=f"exp_{i}") for i in range(n_experts)
        ])
        self.consolidate = nn.Linear(dm * n_experts, dm)
        self.consolidate_norm = nn.LayerNorm(dm)
        # Lightweight attention (Jamba-inspired, 1:3 ratio of attn:total passes)
        self.attn = LightweightAttention(dm)
        self.register_buffer("prev_route", torch.zeros(n_experts))
        self.inertia = 0.4
        self.loser_inhibit = INHIBIT_LR
        self.topk = topk
        self.router_bias = nn.Parameter(torch.zeros(n_experts))
        self.expert_norm = nn.LayerNorm(dm)

    def forward(self, x):
        bsz = x.shape[0]
        device = x.device
        emb = self.embed_norm(self.embed(x))
        h = torch.zeros(bsz, self.embed.weight.shape[1], device=device)
        h_seq = []

        self._router_qloss = 0.0
        self._loss_count = 0

        for t in range(x.shape[1]):
            x_emb = emb[:, t, :]
            self.prev_route.zero_()
            h_prev = h
            h_states = [torch.zeros(bsz, self.embed.weight.shape[1], device=device) for _ in self.experts]
            route_logits = []

            for depth in range(3):
                route_raw = self.router(x_emb) + self.router_bias
                if self.training and self.route_noise > 0:
                    route_raw = route_raw + torch.randn_like(route_raw) * self.route_noise
                route_logits.append(route_raw)
                route_smooth = self.inertia * self.prev_route + (1 - self.inertia) * route_raw
                route_weights = torch.softmax(route_smooth, dim=-1)
                self._last_route_entropy = -(route_weights * torch.log(route_weights.clamp(min=1e-10))).sum(-1).mean().item()
                self.prev_route = route_weights.detach()

                h_exps = []
                h_fasts = []
                for i, exp in enumerate(self.experts):
                    h_out, h_fast = exp.step(h_states[i], x_emb)
                    h_states[i] = h_fast.detach()
                    h_exps.append(h_out)
                    h_fasts.append(h_fast)

                h_exps = [self.expert_norm(h) for h in h_exps]
                if self.topk > 0 and self.topk < len(self.experts):
                    _, indices = route_weights.topk(self.topk, dim=-1)
                    mask = torch.zeros_like(route_weights).scatter_(1, indices, 1)
                    h_exps = [h * mask[:, i:i+1] for i, h in enumerate(h_exps)]

                h_stack = torch.cat(h_exps, dim=-1)
                h_new = self.consolidate_norm(self.consolidate(h_stack))

                if depth > 0 and (h_new - h_prev).norm().item() / (h_prev.norm().item() + 1e-8) < CONV_THRESH:
                    break
                h_prev = h_new
                h = h_new

                if self.training:
                    h_stack_exps = torch.stack(h_exps)
                    with torch.no_grad():
                        err = (h_stack_exps - h_new.unsqueeze(0)).norm(dim=-1)
                        target = (-err).softmax(dim=0).T
                    log_prob = route_raw.log_softmax(-1)
                    self._router_qloss += -(target * log_prob).sum(-1).mean()
                    self._loss_count += 1

            # Hebbian update
            for i, exp in enumerate(self.experts):
                hf = h_fasts[i]
                B = hf.shape[0]
                winner_idx = route_weights.argmax(dim=-1)
                for b in range(B):
                    if winner_idx[b] == i:
                        p = exp.patterns / exp.patterns.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                        score = p @ hf[b]
                        winner = score.argmax()
                        p[winner] += HEBB_LR * hf[b]
                        p = F.normalize(p, dim=-1)
                        if self.training:
                            with torch.no_grad():
                                sim = score / (score.norm() + 1e-10)
                                loser_mask = sim > 0.9
                                loser_idx = torch.where(loser_mask)[0]
                                for li in loser_idx:
                                    if li != winner:
                                        diff = p.data[li] - p.data[winner]
                                        p.data[li] -= self.loser_inhibit * diff

            h_seq.append(h.unsqueeze(0))

        # Cross-token attention over full sequence (Jamba-inspired)
        h_seq = torch.cat(h_seq, dim=0)  # [S, B, D]
        h_seq = h_seq.transpose(0, 1)  # [B, S, D]
        h_seq = self.attn(h_seq) + h_seq  # residual attention
        logits = self.head(h_seq)

        # Auxiliary losses
        if self.training and route_logits:
            logits_flat = torch.stack(route_logits, dim=0).transpose(0, 1).reshape(-1, len(self.experts))
            p = torch.softmax(logits_flat, dim=-1)
            f = p.mean(dim=0)
            aux_loss = self.aux_loss_weight * len(self.experts) * (f * p.mean(0)).sum()
            log_z = logits_flat.logsumexp(-1)
            z_loss = (log_z ** 2).mean() * 1e-4
            router_z_loss = 1e-2 * (logits_flat ** 2).mean()
            self._last_aux_loss = aux_loss + z_loss + router_z_loss
            if self._loss_count > 0:
                self._router_qloss /= self._loss_count
                aux_total = self._router_qloss * 0.1
                self._last_aux_loss += max(0.0, aux_total)
        else:
            self._last_aux_loss = 0.0

        return logits

    def finish_training_step(self):
        self._aux_step = getattr(self, '_aux_step', 0) + 1
