"""
MoHE: Mixture of Hebbian Experts with Depth-of-Thought.
"""
import torch, torch.nn as nn

CONV_THRESH = 0.05
LR = 1e-4
INHIBIT_LR = 0.1

try:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rina.kernels import fused_all_experts as _fused_all_experts
except Exception:
    _fused_all_experts = None

try:
    from rina.kernels.train import FusedExpertFunction as _FEF, pack_weights as _pack_weights
    _train_fuse = True
except Exception:
    _train_fuse = False


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
        self._packed_weights = None
        self._packed_P = None

    def forward(self, x, max_depth=1):
        bsz, seq_len = x.shape
        dm = self.experts[0].patterns.shape[1]
        emb = self.embed_norm(self.embed(x))
        h = torch.zeros(bsz, dm, device=x.device)
        h_seq = []

        if self.training and _train_fuse:
            gw, gb, fw, fb, pw_, pb_, _, fmw, fmb, nw, nb, sw, sb = _pack_weights(self)
        for t in range(seq_len):
            x_emb = emb[:, t, :]
            self.prev_route.zero_()
            h_prev = h
            if self.training and _train_fuse:
                P = torch.stack([e.patterns.T @ e.patterns for e in self.experts])

            for depth in range(max_depth):
                combined = torch.cat([h, x_emb], dim=-1)

                route_raw = self.router(combined)
                route_smooth = self.inertia * self.prev_route + (1 - self.inertia) * route_raw
                route_weights = torch.softmax(route_smooth, dim=-1)
                self.prev_route = route_weights.detach()

                h_exps = []
                h_fasts = []
                if self.training and _train_fuse:
                    h_out_pk, h_fast_pk = _FEF.apply(
                        h, x_emb, gw, gb, fw, fb, pw_, pb_, P, fmw, fmb, nw, nb, sw, sb)
                    h_exps = [h_out_pk[i] for i in range(len(self.experts))]
                    h_fasts = [h_fast_pk[i] for i in range(len(self.experts))]
                else:
                    h_out_packed, h_fast_packed = _fused_all_experts(h, x_emb, self)
                    h_exps = [h_out_packed[i] for i in range(len(self.experts))]
                    h_fasts = [h_fast_packed[i] for i in range(len(self.experts))]

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

            h_seq.append(h)

        # Batched head: [bs, seq, dm] → [bs*seq, dm] → head → [bs*seq, vocab] → [bs, seq, vocab]
        h_flat = torch.stack(h_seq, dim=1).reshape(-1, dm)
        logits = torch.clamp(self.head(self.state_norm(h_flat)) / 3, -20, 20)
        return logits.reshape(bsz, seq_len, -1)

    def finish_training_step(self):
        """Call after loss.backward() to compute expert parameter gradients."""
        if _train_fuse:
            from rina.kernels.train import compute_param_grads, apply_param_grads
            grads = compute_param_grads()
            if grads is not None:
                apply_param_grads(self, grads)
