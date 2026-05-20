import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.hash_slot import HashSlot


class CANNCell(nn.Module):
    """CANN dynamics with optional external input."""
    def __init__(self, d_model, n_patterns=2048, beta=1.0, n_iter=3):
        super().__init__()
        self.d_model = d_model
        self.n_iter = n_iter
        self.beta = beta
        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.in_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _retrieve(self, h):
        bsz = h.shape[0]
        pat = self.patterns.unsqueeze(0).expand(bsz, -1, -1)
        xi = h.unsqueeze(1)
        for _ in range(self.n_iter):
            scores = xi @ pat.transpose(1, 2) * self.beta
            attn = F.softmax(scores, dim=-1)
            xi = attn @ pat
            xi = self.norm(xi)
        return xi.squeeze(1)

    def forward(self, h, x):
        combined = self.in_proj(torch.cat([h, x], dim=-1))
        combined = self.norm(combined)
        return self._retrieve(combined)


class RINAModel(nn.Module):
    """
    RINA v3: CANN + Exact HashSlot + logit bias injection.

    Flow per token t:
      1. Current token x_t → query HashSlot (content-based)
      2. If hit: bias final logits toward the retrieved value token
      3. CANN state update: h = CANN(h, embed(x_t))
      4. Predict next token from h (+slot logit bias if applicable)
    """
    def __init__(self, vocab_size, d_model=256, n_patterns=2048,
                 beta=1.0, n_iter=3, n_slots=4096, error_threshold=0.5):
        super().__init__()
        self.d_model = d_model
        self.error_threshold = error_threshold
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cann = CANNCell(d_model, n_patterns=n_patterns, beta=beta, n_iter=n_iter)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.slot = HashSlot(capacity=n_slots)
        self.slot_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, return_state=False):
        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []

        for t in range(seq_len):
            h = self.cann(h, emb[:, t, :])
            h_norm = self.state_norm(h)
            logit = self.head(h_norm)

            token_id = x[0, t].item()
            slot_val = self.slot.lookup(token_id)
            if slot_val is not None:
                logit[:, slot_val] += self.slot_bias[slot_val]

            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        if return_state:
            return logits, h
        return logits

    def forward_with_slot_writes(self, x):
        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []

        for t in range(seq_len):
            h = self.cann(h, emb[:, t, :])
            h_norm = self.state_norm(h)
            logit = self.head(h_norm)

            token_id = x[0, t].item()
            slot_val = self.slot.lookup(token_id)
            if slot_val is not None:
                logit[:, slot_val] += self.slot_bias[slot_val]

            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        return logits

    @torch.no_grad()
    def process_and_learn(self, x, y, losses_per_step):
        """
        Process sequence and write to slot based on per-step prediction errors.

        x: (batch, seq_len) input tokens
        y: (batch, seq_len) target tokens (shifted by 1)
        losses_per_step: (batch, seq_len) loss values for each position

        Writes: for each position t where loss[t] > threshold,
                write (x[t], t, y[t]) to slot
        """
        for b in range(x.shape[0]):
            for t in range(x.shape[1] - 1):
                if losses_per_step[b, t] > self.error_threshold:
                    self.slot.insert(
                        token_id=x[b, t].item(),
                        position=t,
                        value_token_id=y[b, t].item()
                    )
        self.slot.tick()
