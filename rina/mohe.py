"""
MoHE: Mixture of Hebbian Experts with Depth-of-Thought.
"""
import torch, torch.nn as nn

CONV_THRESH = 0.05
LR = 1e-4
HEBB_LR = 0.05
INHIBIT_LR = 0.5

try:
    from rina.kernels import FusedLightFunction as _FusedLightFunction
    _use_light = True
except Exception:
    _use_light = False


class ExpertCell(nn.Module):
    """Single Hebbian expert: fast SSM + slow linear field."""
    def __init__(self, dm, np_, name=""):
        super().__init__()
        self.name = name
        self.gate_a = nn.Linear(dm, dm)
        self.gate_b = nn.Linear(dm, dm)
        self.proj_in = nn.Linear(dm, dm)
        self.slow_gate = nn.Linear(dm * 2, 1)
        self.field_mix = nn.Linear(dm, dm)
        self.norm = nn.LayerNorm(dm)
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)

    def forward(self, h, x_emb):
        a = torch.sigmoid(self.gate_a(x_emb))
        b = torch.sigmoid(self.gate_b(x_emb))
        xp = self.proj_in(x_emb)
        h_fast = a * h + b * xp
        return self._attract(h, h_fast, x_emb), h_fast

    def _attract(self, h, h_fast, x_emb):
        """Attractor path only (gate_a/b already computed)."""
        P = self.patterns.T @ self.patterns
        field = h_fast @ P
        field = self.field_mix(field)
        field = self.norm(field)
        gate = torch.sigmoid(self.slow_gate(torch.cat([h, x_emb], dim=-1)))
        return h_fast + gate * field * 0.1


class MoHE(nn.Module):
    """Multi-layer MoHE with Depth-of-Thought."""
    def __init__(self, vocab, dm, np_, n_experts=4, aux_loss_weight=0.1,
                 route_noise=0.0, expert_dropout=0.0, topk=0):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.route_noise = route_noise
        self.expert_dropout = expert_dropout
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
            ExpertCell(dm, np_, name=f"exp_{i}") for i in range(n_experts)
        ])
        self.consolidate = nn.Linear(dm * n_experts, dm)
        self.consolidate_norm = nn.LayerNorm(dm)

        self.topk = topk
        self.router_bias = nn.Parameter(torch.zeros(n_experts))
        self.expert_norm = nn.LayerNorm(dm)
        self.bias_lr = 1.0
        self.register_buffer("_batch_counts", torch.zeros(n_experts))
        self.register_buffer("_batch_total", torch.zeros(1))
        self.register_buffer("prev_route", torch.zeros(n_experts))
        self.inertia = 0.4
        self.loser_inhibit = INHIBIT_LR

    def forward(self, x, max_depth=1):
        bsz, seq_len = x.shape
        dm = self.experts[0].patterns.shape[1]
        emb = self.embed_norm(self.embed(x))
        h = torch.zeros(bsz, dm, device=x.device)
        h_seq = []
        self._conv_total = 0
        self._conv_hits = 0
        self._cap_fired = 0
        route_logits = []
        self._router_qloss = 0.0
        self._diversity_loss = 0.0
        self._loss_count = 0

        # Batch gate precomputation for entire sequence
        a_stacked = torch.stack([torch.sigmoid(exp.gate_a(emb)) for exp in self.experts])
        b_stacked = torch.stack([torch.sigmoid(exp.gate_b(emb)) for exp in self.experts])
        xp_stacked = torch.stack([exp.proj_in(emb) for exp in self.experts])
        if self.training and _use_light:
            fmw = torch.stack([e.field_mix.weight for e in self.experts])
            fmb = torch.stack([e.field_mix.bias for e in self.experts])
            nw = torch.stack([e.norm.weight for e in self.experts])
            nb = torch.stack([e.norm.bias for e in self.experts])
            sw = torch.stack([e.slow_gate.weight.squeeze(0) for e in self.experts])
            sb = torch.stack([e.slow_gate.bias.squeeze(0) for e in self.experts])
        for t in range(seq_len):
            x_emb = emb[:, t, :]
            self.prev_route.zero_()
            h_prev = h
            if self.training and _use_light:
                P = torch.stack([e.patterns.T @ e.patterns for e in self.experts])

            for depth in range(max_depth):
                self._conv_total += 1

                route_raw = self.router(x_emb) + self.router_bias
                if self.training and self.route_noise > 0:
                    route_raw = route_raw + torch.randn_like(route_raw) * self.route_noise
                route_logits.append(route_raw)
                route_smooth = self.inertia * self.prev_route + (1 - self.inertia) * route_raw
                route_weights = torch.softmax(route_smooth, dim=-1)
                if self.training:
                    usage_ratio = route_weights.max(-1).values / route_weights.min(-1).values.clamp(min=1e-10)
                    if usage_ratio.median() > 10:
                        self._cap_fired += 1
                self._last_route_entropy = -(route_weights * torch.log(route_weights.clamp(min=1e-10))).sum(-1).mean().item()
                self.prev_route = route_weights.detach()

                h_exps = []
                h_fasts = []
                if self.training and _use_light:
                    h_fast = (a_stacked[:, :, t, :] * h.unsqueeze(0) +
                              b_stacked[:, :, t, :] * xp_stacked[:, :, t, :])
                    h_out_pk, _ = _FusedLightFunction.apply(
                        h_fast, h, x_emb, P, fmw, fmb, nw, nb, sw, sb)
                    h_exps = [h_out_pk[i] for i in range(len(self.experts))]
                    h_fasts = [h_fast[i] for i in range(len(self.experts))]
                else:
                    for i, exp in enumerate(self.experts):
                        h_fast_t = a_stacked[i, :, t, :] * h + b_stacked[i, :, t, :] * xp_stacked[i, :, t, :]
                        h_out = exp._attract(h, h_fast_t, x_emb)
                        h_exps.append(h_out)
                        h_fasts.append(h_fast_t)

                if self.training and self.expert_dropout > 0:
                    keep = torch.bernoulli(torch.full((len(self.experts),),
                        1 - self.expert_dropout, device=h.device)).unsqueeze(1).unsqueeze(2)
                    h_exps = [h * k for h, k in zip(h_exps, keep)]

                h_exps = [self.expert_norm(h) for h in h_exps]

                if self.topk > 0 and self.topk < len(self.experts):
                    _, indices = route_weights.topk(self.topk, dim=-1)
                    mask = torch.zeros_like(route_weights).scatter_(1, indices, 1)
                    h_exps = [h * mask[:, i:i+1] for i, h in enumerate(h_exps)]

                h_stack = torch.cat(h_exps, dim=-1)
                h_new = self.consolidate_norm(self.consolidate(h_stack))

                if self.training:
                    h_stack_exps = torch.stack(h_exps)
                    with torch.no_grad():
                        err = (h_stack_exps - h_new.unsqueeze(0)).norm(dim=-1)
                        target = (-err).softmax(dim=0).T
                    log_prob = route_raw.log_softmax(-1)
                    self._router_qloss += -(target * log_prob).sum(-1).mean()
                    h_avg = h_stack_exps.mean(dim=0, keepdim=True)
                    self._diversity_loss += -((h_stack_exps - h_avg) ** 2).mean()
                    self._loss_count += 1

                if depth > 0 and (h_new - h_prev).norm().item() / (h_prev.norm().item() + 1e-8) < CONV_THRESH:
                    self._conv_hits += 1
                    break
                h_prev = h_new
                h = h_new

            if self.training:
                winner_idx = route_weights.argmax(dim=-1)
                h = torch.nan_to_num(h)
                hf_stack = torch.stack(h_fasts)
                # Winner's h_fast for each batch element (used in loser push)
                winner_hf_all = hf_stack[winner_idx, torch.arange(bsz, device=h.device)]

                for i, expert in enumerate(self.experts):
                    # ── Winner update ──
                    mask = winner_idx == i
                    h_win = h_fasts[i][mask]
                    if h_win.numel() == 0: continue
                    with torch.no_grad():
                        k = (h_win @ expert.patterns.T).argmax(dim=-1)
                        delta = h_win - expert.patterns[k]
                        expert.patterns.data.index_add_(0, k, HEBB_LR * delta)

                    # ── Loser update ──
                    loser_mask = winner_idx != i
                    h_lose = winner_hf_all[loser_mask]
                    if h_lose.numel() == 0: continue
                    with torch.no_grad():
                        k = (h_lose @ expert.patterns.T).argmax(dim=-1)
                        delta = h_lose - expert.patterns[k]
                        expert.patterns.data.index_add_(0, k,
                            -self.loser_inhibit * HEBB_LR * delta)

            h_seq.append(h)

        if self.training:
            with torch.no_grad():
                for exp in self.experts:
                    norms = exp.patterns.data.norm(dim=1, keepdim=True)
                    exp.patterns.data.div_(norms.clamp(min=1e-8))

        self._conv_rate = self._conv_hits / max(self._conv_total, 1)
        self._cap_rate = self._cap_fired / max(len(route_logits), 1)

        if self.training and route_logits:
            logits_stack = torch.stack(route_logits)  # [T*B, ne]
            logits_flat = logits_stack.view(-1, len(self.experts))
            p = torch.softmax(logits_flat, dim=-1)
            f = torch.zeros(len(self.experts), device=logits_flat.device)
            for i in range(len(self.experts)):
                f[i] = (logits_flat.argmax(-1) == i).float().mean()
            aux_loss = self.aux_loss_weight * len(self.experts) * (f * p.mean(0)).sum()
            cap_penalty = 1.0 * (p.mean(0) - 0.6).clamp(min=0).pow(2).sum()
            aux_loss = aux_loss + cap_penalty
            log_z = logits_flat.logsumexp(-1)
            z_loss = (log_z ** 2).mean() * 1e-4
            self._last_aux_loss = aux_loss + z_loss
            if self._loss_count > 0:
                self._router_qloss /= self._loss_count
                self._diversity_loss /= self._loss_count
                aux_total = self._router_qloss * 0.1 + self._diversity_loss * 0.4
                self._last_aux_loss += max(0.0, aux_total)
            self._gate_ratio = (p.max(-1).values / p.min(-1).values.clamp(min=1e-10)).median().item()
            # Router bias adjustment
            for e in range(len(self.experts)):
                count = (logits_flat.argmax(-1) == e).float().sum().item()
                target = logits_flat.size(0) / len(self.experts)
                if count > target:
                    self.router_bias.data[e] -= self.bias_lr
                elif count < target:
                    self.router_bias.data[e] += self.bias_lr
            self.router_bias.data.clamp_(-2, 2)
        else:
            self._last_aux_loss = 0.0

        # Batched head: [bs, seq, dm] → [bs*seq, dm] → head → [bs*seq, vocab] → [bs, seq, vocab]
        h_flat = torch.stack(h_seq, dim=1).reshape(-1, dm)
        logits = torch.clamp(self.head(self.state_norm(h_flat)), -20, 20)
        return logits.reshape(bsz, seq_len, -1)

    def finish_training_step(self):
        """No-op: all gradients flow through autograd automatically."""
