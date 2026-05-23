import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSNNCell(nn.Module):
    def __init__(self, d_model, n_patterns=4096, beta=1.0,
                 error_threshold=0.5, attract_every=1,
                 hebbian_lr=0.01, hebbian_decay=0.999,
                 inhibition_threshold=0.0, pattern_rank=0):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.register_buffer("error_threshold", torch.tensor([error_threshold]))
        self.register_buffer("inhibition_threshold", torch.tensor([inhibition_threshold]))
        self.rank = pattern_rank

        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.gate_alpha = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        if pattern_rank > 0 and pattern_rank < n_patterns:
            self.U = nn.Parameter(torch.randn(n_patterns, pattern_rank) * 0.02)
            self.V = nn.Parameter(torch.randn(pattern_rank, d_model) * 0.02 / (pattern_rank ** 0.5))
            self.patterns = None
        else:
            self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
            self.U = None
            self.V = None
        self.register_buffer("beta_t", torch.tensor([beta]))

        self.register_buffer("att_calls", torch.zeros(1))
        self.register_buffer("total_steps", torch.zeros(1))
        self.register_buffer("hebbian_updates", torch.zeros(1))
        self.reverse_gating = False

    def forward(self, h, x, step=0):
        bsz, dm = h.shape
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        xp = self.proj_in(x)
        h_ssm = a * h + b * xp

        h_pred = h.detach()
        error = (h_ssm - h_pred).norm(dim=-1) / (h_pred.norm(dim=-1) + 1e-8)

        is_att_step = (step % self.attract_every == (self.attract_every - 1))
        if self.error_threshold[0] < 0:
            need_att = torch.ones(bsz, dtype=torch.bool, device=h.device)
        elif self.reverse_gating:
            need_att = error < self.error_threshold[0]
        else:
            need_att = error > self.error_threshold[0]
        do_att = is_att_step & need_att

        if self.training:
            self.total_steps += bsz
            self.att_calls += do_att.float().sum().detach()

        if do_att.any():
            pat = self.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h_ssm.unsqueeze(1)
            scores = xi @ pat.transpose(1, 2) * self.beta_t[0]
            attn = torch.softmax(scores, dim=-1)
            attracted = (attn @ pat).squeeze(1)
            alpha = torch.sigmoid(self.gate_alpha(combined))
            h_attracted = h_ssm + alpha * (attracted - h_ssm)

            with torch.no_grad():
                k_pred = scores.argmax(dim=-1).squeeze(-1)
                lr = self.hebbian_lr / (1.0 + error)
                lr = lr.clamp(max=self.hebbian_lr)

                active = do_att.nonzero(as_tuple=True)[0]
                if len(active) > 0:
                    pk = k_pred[active]
                    lr_active = lr[active].unsqueeze(-1)
                    dh = h_attracted[active] - self.patterns[pk]

                    self.patterns.data.index_add_(0, pk, lr_active * dh)
                    for upk in pk.unique().tolist():
                        self.patterns.data[upk] *= self.hebbian_decay

                    if self.inhibition_threshold[0] > 0:
                        p_norm = F.normalize(self.patterns, dim=-1)
                        for upk in pk.unique().tolist():
                            sim = p_norm @ p_norm[upk]
                            neighbor_mask = (sim > self.inhibition_threshold[0])
                            neighbor_mask[upk] = False
                            if neighbor_mask.any():
                                mask = (pk == upk)
                                avg_dh = dh[mask].mean(dim=0)
                                repulse = -lr_active[mask].mean() * 0.5 * avg_dh.unsqueeze(0)
                                n_idx = neighbor_mask.nonzero(as_tuple=True)[0]
                                self.patterns.data.index_add_(0, n_idx,
                                    repulse.expand(len(n_idx), -1))
                                self.patterns.data[n_idx] *= self.hebbian_decay

                self.hebbian_updates += do_att.float().sum().detach()

            mask = do_att.float().unsqueeze(-1)
            h_new = mask * h_attracted + (1.0 - mask) * h_ssm
        else:
            h_new = h_ssm

        return self.norm(h_new)

    @property
    def att_rate(self):
        if self.total_steps.item() == 0:
            return 1.0
        return (self.att_calls / self.total_steps).item()

    @property
    def hebb_rate(self):
        if self.total_steps.item() == 0:
            return 0.0
        return (self.hebbian_updates / self.total_steps).item()
