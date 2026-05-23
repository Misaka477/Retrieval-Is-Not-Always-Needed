"""
MoHE: Mixture of Hebbian Experts with Depth-of-Thought.
"""
import torch, torch.nn as nn

CONV_THRESH = 0.05
LR = 1e-4
INHIBIT_LR = 0.1


class ExpertCell(nn.Module):
    """Single Hebbian expert: fast SSM + slow linear field."""
    def __init__(self, dm, np_, name=""):
        super().__init__()
        self.name = name
        self.gate_a = nn.Linear(dm * 2, dm)
        self.gate_b = nn.Linear(dm * 2, dm)
        self.proj_in = nn.Linear(dm, dm)
        self.slow_gate = nn.Linear(dm * 2, 1)
        self.field_mix = nn.Linear(dm, dm)
        self.norm = nn.LayerNorm(dm)
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)

    def forward(self, h, x_emb):
        combined = torch.cat([h, x_emb], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        xp = self.proj_in(x_emb)
        h_fast = a * h + b * xp
        P = self.patterns.T @ self.patterns
        field = h_fast @ P
        field = self.field_mix(field)
        field = self.norm(field)
        gate = torch.sigmoid(self.slow_gate(combined))
        return h_fast + gate * field * 0.1, h_fast


class MoHE(nn.Module):
    """Multi-layer MoHE with Depth-of-Thought."""
    def __init__(self, vocab, dm, np_, n_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dm)
        self.embed.weight.data.normal_(0, 0.05)
        self.embed_norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab)
        self.head.weight.data.normal_(0, 0.001)
        self.head.bias.data.fill_(-10.8)
        self.head_bias = -10.8
        self.state_norm = nn.LayerNorm(dm)

        self.router = nn.Linear(dm * 2, n_experts)
        self.experts = nn.ModuleList([
            ExpertCell(dm, np_, name=f"exp_{i}") for i in range(n_experts)
        ])
        self.consolidate = nn.Linear(dm * n_experts, dm)
        self.consolidate_norm = nn.LayerNorm(dm)

        self.register_buffer("prev_route", torch.zeros(4))
        self.inertia = 0.7
        self.loser_inhibit = INHIBIT_LR

    def forward(self, x, max_depth=1):
        bsz, seq_len = x.shape
        emb = self.embed_norm(self.embed(x))
        h = torch.zeros(bsz, self.experts[0].patterns.shape[1], device=x.device)
        logits = []

        for t in range(seq_len):
            x_emb = emb[:, t, :]
            self.prev_route.zero_()
            h_prev = h

            for depth in range(max_depth):
                combined = torch.cat([h, x_emb], dim=-1)

                route_raw = self.router(combined)
                route_smooth = self.inertia * self.prev_route + (1 - self.inertia) * route_raw
                route_weights = torch.softmax(route_smooth, dim=-1)
                self.prev_route = route_weights.detach()

                h_exps = []
                h_fasts = []
                for i, expert in enumerate(self.experts):
                    h_out, h_fast = expert(h, x_emb)
                    h_exps.append(h_out)
                    h_fasts.append(h_fast)

                h_stack = torch.cat(h_exps, dim=-1)
                h_new = self.consolidate_norm(self.consolidate(h_stack))

                if depth > 0 and (h_new - h_prev).norm().item() / (h_prev.norm().item() + 1e-8) < CONV_THRESH:
                    break
                h_prev = h_new
                h = h_new

            if self.training:
                winner_idx = route_weights.argmax(dim=-1)
                h = torch.nan_to_num(h)

                for i, expert in enumerate(self.experts):
                    mask = winner_idx == i
                    if mask.any():
                        h_target = torch.nan_to_num(h[mask])
                        h_fast_i = torch.nan_to_num(h_fasts[i][mask])
                        with torch.no_grad():
                            diff = (h_fast_i - h_target).norm(dim=-1) / (h_target.norm(dim=-1) + 1e-8)
                            scores = h_target.unsqueeze(1) @ expert.patterns.T.unsqueeze(0)
                            k = scores.squeeze(1).argmax(dim=-1)
                            delta = h_target - expert.patterns[k]
                            expert.patterns.data.index_add_(0, k,
                                LR / (1 + diff.unsqueeze(-1)) * delta)
                    else:
                        winner_h = torch.nan_to_num(h[winner_idx != i])
                        if len(winner_h) > 0:
                            with torch.no_grad():
                                scores = winner_h.unsqueeze(1) @ expert.patterns.T.unsqueeze(0)
                                k = scores.squeeze(1).argmax(dim=-1)
                                delta = winner_h - expert.patterns[k]
                                expert.patterns.data.index_add_(0, k,
                                    -self.loser_inhibit * LR * delta)

            logits.append(torch.clamp(self.head(self.state_norm(h)) / 3, -20, 20))

        return torch.stack(logits, dim=1)
