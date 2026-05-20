import torch
import torch.nn as nn
import torch.nn.functional as F


class RINACell(nn.Module):
    """
    Unified attractor field with dual basins: normal + protected.

    Core dynamics: h_t = RINA(h_{t-1}, x_t, pred_error)

    The field has two types of attractor basins:
      - Normal (N):    shallow, constantly overwritten by new inputs
      - Protected (P): deep, only updated when pred_error is high

    Retrieval = state iteratively pulled toward nearest attractor
    (always, regardless of type).

    Writing to protected basins is gated by prediction error:
      - High pred_error + high novelty → allocate protected slot
      - Protected slots are LRU-evicted when full
    """
    def __init__(self, d_model, n_normal=2048, n_protected=128, beta=1.0, n_iter=3):
        super().__init__()
        self.d_model = d_model
        self.n_iter = n_iter
        self.beta = beta

        self.normal = nn.Parameter(torch.randn(n_normal, d_model) * 0.02)
        self.protected = nn.Parameter(torch.randn(n_protected, d_model) * 0.02)

        self.in_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.prot_gate = nn.Linear(d_model * 2, 1)

        self.register_buffer("usage", torch.zeros(n_protected))
        self.register_buffer("age", torch.zeros(n_protected, dtype=torch.long))

        self.prot_gate_bias = nn.Parameter(torch.tensor(0.0))

    def _retrieve(self, h):
        """Pull state toward nearest attractor in the unified field."""
        bsz = h.shape[0]
        patterns = torch.cat([self.normal, self.protected], dim=0)
        pat = patterns.unsqueeze(0).expand(bsz, -1, -1)
        xi = h.unsqueeze(1)

        for _ in range(self.n_iter):
            scores = xi @ pat.transpose(1, 2) * self.beta
            attn = F.softmax(scores, dim=-1)
            xi = attn @ pat
            xi = self.norm(xi)

        return xi.squeeze(1)

    def forward(self, h, x, pred_error=None):
        """
        h: (batch, d_model) - previous state
        x: (batch, d_model) - input embedding
        pred_error: (batch,) or None - prediction error signal

        returns: (batch, d_model) - new state
        """
        bsz = h.shape[0]
        combined = self.in_proj(torch.cat([h, x], dim=-1))
        combined = self.norm(combined)
        h_new = self._retrieve(combined)

        if pred_error is not None and self.training:
            gate = torch.sigmoid(self.prot_gate(torch.cat([h_new, x], dim=-1)).squeeze(-1))
            write_signal = gate * pred_error
            self._update_protected(h_new, write_signal)

        return h_new

    @torch.no_grad()
    def _update_protected(self, h_new, write_signal):
        """Write high-error states into protected basins (LRU)."""
        n_prot = self.protected.shape[0]
        self.age += 1

        for b in range(h_new.shape[0]):
            w = write_signal[b].item()
            if w < 0.5:
                continue

            vec = h_new[b].detach()
            sims = F.cosine_similarity(
                self.protected, vec.unsqueeze(0), dim=-1
            )

            best_sim, best_idx = sims.max(dim=0)
            if best_sim > 0.9:
                self.protected[best_idx] = 0.9 * self.protected[best_idx] + 0.1 * vec
                self.age[best_idx] = 0
                self.usage[best_idx] += 1
            else:
                oldest = self.age.argmax().item()
                self.protected[oldest] = vec
                self.age[oldest] = 0
                self.usage[oldest] = 1


class RINASimpleModel(nn.Module):
    """
    RINA prototype with dual attractor field.
    """
    def __init__(self, vocab_size, d_model=256, n_normal=2048, n_protected=128,
                 beta=1.0, n_iter=3, error_threshold=0.5):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rina = RINACell(d_model, n_normal=n_normal, n_protected=n_protected,
                             beta=beta, n_iter=n_iter)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.error_threshold = error_threshold

    def forward(self, x, return_state=False):
        bsz, seq_len = x.shape
        emb = self.embed(x)

        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []

        for t in range(seq_len):
            h = self.rina(h, emb[:, t, :])
            h_norm = self.state_norm(h)
            logit = self.head(h_norm)
            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        if return_state:
            return logits, h
        return logits

    def forward_with_error(self, x):
        """Train with prediction-error-driven protected slot allocation."""
        bsz, seq_len = x.shape
        emb = self.embed(x)

        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []
        h_states = []

        for t in range(seq_len):
            h = self.rina(h, emb[:, t, :])
            h_states.append(h)
            h_norm = self.state_norm(h)
            logit = self.head(h_norm)
            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        return logits, torch.stack(h_states, dim=1)
