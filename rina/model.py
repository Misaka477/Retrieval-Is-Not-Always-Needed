import torch
import torch.nn as nn
import torch.nn.functional as F

from rina.cell import TemporalSNNCell


class TemporalSNNModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=4096,
                 beta=1.0, attract_every=1, error_threshold=0.5,
                 hebbian_lr=0.0, inhibition_threshold=0.0, pattern_rank=0,
                 n_slots=0):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = TemporalSNNCell(d_model, n_patterns=n_patterns,
                                     beta=beta, attract_every=attract_every,
                                     error_threshold=error_threshold,
                                     hebbian_lr=hebbian_lr,
                                     inhibition_threshold=inhibition_threshold,
                                     pattern_rank=pattern_rank)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)

        self.n_slots = n_slots
        if n_slots > 0:
            self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))
            self.slot_proj = nn.Linear(d_model, d_model)

    def slot_write(self, key_id, value_id):
        with torch.no_grad():
            v = self.slot_proj(self.embed(torch.tensor([value_id], device=self.slot_table.device)))
            self.slot_table[key_id] = v.squeeze(0)

    def forward(self, x, return_states=False):
        bsz, seq_len = x.shape
        dm = self.d_model
        emb = self.embed(x)
        h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        states = [h.clone()] if return_states else None
        for t in range(seq_len):
            h_in = h + self.slot_table[x[:, t]] if self.n_slots > 0 else h
            h = self.cell(h_in, emb[:, t, :], step=t)
            if t == seq_len - 1:
                pat = self.cell.patterns.unsqueeze(0).expand(bsz, -1, -1) if self.cell.patterns is not None else ((self.cell.U @ self.cell.V).unsqueeze(0).expand(bsz, -1, -1))
                xi = h.unsqueeze(1)
                scores = xi @ pat.transpose(1, 2) * self.cell.beta_t[0]
                attn = torch.softmax(scores, dim=-1)
                attracted = (attn @ pat).squeeze(1)
                combined_last = torch.cat([h, emb[:, -1, :]], dim=-1)
                alpha = torch.sigmoid(self.cell.gate_alpha(combined_last))
                h = h + alpha * (attracted - h)
                h = self.cell.norm(h)
            logits.append(self.head(self.state_norm(h)))
            if return_states:
                states.append(h.clone())
        out = torch.stack(logits, dim=1)
        if return_states:
            return out, torch.stack(states, dim=1)
        return out

    @torch.no_grad()
    def generate(self, prompt_ids, max_len=128, temperature=0.8, top_k=20):
        self.eval()
        x = torch.tensor([prompt_ids], dtype=torch.long, device=next(self.parameters()).device)
        for _ in range(max_len):
            logits = self(x)
            logit = logits[0, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logit, top_k)
                logit[logit < v[-1]] = float("-inf")
            probs = F.softmax(logit, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            x = torch.cat([x[:, -63:], torch.tensor([[next_id]], device=x.device)], dim=1)
            yield next_id

    def get_att_rate(self):
        return self.cell.att_rate

    def get_hebb_rate(self):
        return self.cell.hebb_rate
