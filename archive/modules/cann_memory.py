import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CANNCell(nn.Module):
    """
    Single-step CANN state update.

    Core dynamics: h_t = CANN(h_{t-1}, x_t)
    where CANN is a continuous attractor that:
      - Maintains a state vector h (position in the neural field)
      - Uses stored patterns as attractor basins
      - Iteratively moves h toward the closest attractor

    SSM-style coupling: CANN replaces the SSM's linear A matrix
    with attractor dynamics, giving permanent memory.
    """
    def __init__(self, d_model, n_patterns=4096, beta=1.0, n_iter=5):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.beta = beta
        self.n_iter = n_iter

        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.in_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _retrieve(self, h):
        """
        Move state toward nearest attractor.
        h: (batch, d_model)
        returns: (batch, d_model)
        """
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
        """
        h: (batch, d_model) - previous state
        x: (batch, d_model) - input token embedding
        returns: (batch, d_model) - new state
        """
        combined = self.in_proj(torch.cat([h, x], dim=-1))
        combined = self.norm(combined)
        h_new = self._retrieve(combined)
        return h_new


class CANNSimpleModel(nn.Module):
    """
    Minimal RINA prototype: Embedding - CANN state - Prediction head.

    Processes a sequence token by token, maintaining a CANN state.
    For each position, the state is used to predict the next token.

    This tests the core hypothesis: can CANN attractor dynamics
    maintain information across very long sequences?
    """
    def __init__(self, vocab_size, d_model=256, n_patterns=4096, beta=1.0, n_iter=5):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cann = CANNCell(d_model, n_patterns=n_patterns, beta=beta, n_iter=n_iter)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_state=False):
        """
        x: (batch, seq_len) token ids
        returns: logits (batch, seq_len, vocab_size)
        """
        bsz, seq_len = x.shape
        emb = self.embed(x)

        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []

        for t in range(seq_len):
            h = self.cann(h, emb[:, t, :])
            h_norm = self.state_norm(h)
            logit = self.head(h_norm)
            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        if return_state:
            return logits, h
        return logits

    @torch.no_grad()
    def recall_state(self, x):
        self.eval()
        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.d_model, device=x.device)
        for t in range(seq_len):
            h = self.cann(h, emb[:, t, :])
        return h
