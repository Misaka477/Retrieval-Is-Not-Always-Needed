"""
DEQ-hybrid cell 单测 — 先验证类本身能跑通, 再集成到训练脚本。
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

class DEQHybridCell(torch.nn.Module):
    """attractor 步中 P 冻住 (detach), Hebbian 从 BPTT 图拔出."""
    def __init__(self, d_model, n_patterns=256, beta=1.0, error_threshold=0.5,
                 attract_every=1, hebbian_lr=0.01, hebbian_decay=0.999,
                 inhibition_threshold=0.0):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.register_buffer("error_threshold", torch.tensor([error_threshold]))
        self.register_buffer("inhibition_threshold", torch.tensor([inhibition_threshold]))

        self.gate_a = torch.nn.Linear(d_model * 2, d_model)
        self.gate_b = torch.nn.Linear(d_model * 2, d_model)
        self.gate_alpha = torch.nn.Linear(d_model * 2, d_model)
        self.proj_in = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)

        self.patterns = torch.nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
        self.register_buffer("beta_t", torch.tensor([beta]))

        self.register_buffer("att_calls", torch.zeros(1))
        self.register_buffer("total_steps", torch.zeros(1))
        self.register_buffer("hebbian_updates", torch.zeros(1))

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
        else:
            need_att = error > self.error_threshold[0]
        do_att = is_att_step & need_att

        if self.training:
            self.total_steps += bsz
            self.att_calls += do_att.float().sum().detach()

        if do_att.any():
            # DEQ 核心: attractor 用冻住的 P, 梯度不流过 Hebbian 修改的 P
            P_stable = self.patterns.detach()
            pat = P_stable.unsqueeze(0).expand(bsz, -1, -1)
            xi = h_ssm.unsqueeze(1)
            scores = xi @ pat.transpose(1, 2) * self.beta_t[0]
            attn = torch.softmax(scores, dim=-1)
            attracted = (attn @ pat).squeeze(1)
            alpha_g = torch.sigmoid(self.gate_alpha(combined))
            h_attracted = h_ssm + alpha_g * (attracted - h_ssm)

            # Hebbian: 完全在 no_grad 中, 不进入 BPTT 图
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


# ── 单测 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dm, np_ = 256, 512
cell = DEQHybridCell(dm, n_patterns=np_, beta=0.5, error_threshold=0.5,
                     attract_every=2, hebbian_lr=0.01,
                     inhibition_threshold=0.0).to(device)
print(f"Cell created: d={dm}, np={np_}, params={sum(p.numel() for p in cell.parameters()):,}")

# 单步前向
h = torch.zeros(2, dm, device=device)
x = torch.randn(2, dm, device=device)
out = cell(h, x, step=0)
print(f"Forward: h={out.shape}, norm={out.norm():.4f}")

# 多步序列前向 + backward
h = torch.zeros(2, dm, device=device)
for t in range(4):
    h = cell(h, x, step=t)
loss = h.norm()
loss.backward()
print(f"Backward OK. att_rate={cell.att_rate:.2f}")

# pattern 检查: Hebbian 是否生效
p_norm_before = cell.patterns.norm(dim=-1).mean().item()
print(f"Pattern mean norm: {p_norm_before:.4f}")
print(f"Hebbian updates: {cell.hebbian_updates.item():.0f}")

# gradient check: gate_a 有梯度吗
ga_grad_norm = cell.gate_a.weight.grad.norm().item()
print(f"gate_a grad norm: {ga_grad_norm:.6f} {'[PASS]' if ga_grad_norm > 0 else '[FAIL: no grad]'}")

# ── 集成到序列模型, 跑 mini 训练 ──
print(f"\n── Mini LM training test ──")

class DEQHybridModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=256, beta=0.5,
                 attract_every=2, error_threshold=0.5, hebbian_lr=0.01,
                 inhibition_threshold=0.0):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.cell = DEQHybridCell(d_model, n_patterns=n_patterns, beta=beta,
                                   attract_every=attract_every,
                                   error_threshold=error_threshold,
                                   hebbian_lr=hebbian_lr,
                                   inhibition_threshold=inhibition_threshold)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.state_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        bsz, sl = x.shape; dm = self.d_model
        emb = self.embed(x); h = torch.zeros(bsz, dm, device=x.device)
        logits = []
        for t in range(sl):
            if t < sl - 1:
                h = self.cell(h, emb[:, t, :], step=t)
            else:
                h = self.cell(h, emb[:, t, :], step=t)
                pat = self.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
                xi = h.unsqueeze(1)
                scores = xi @ pat.transpose(1, 2) * self.cell.beta_t[0]
                attn = torch.softmax(scores, dim=-1)
                attracted = (attn @ pat).squeeze(1)
                combined_last = torch.cat([h, emb[:, -1, :]], dim=-1)
                alpha = torch.sigmoid(self.cell.gate_alpha(combined_last))
                h = h + alpha * (attracted - h)
                h = self.cell.norm(h)
            logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)

    def get_att_rate(self):
        return self.cell.att_rate

# 合成数据 mini 训练
V = 256
model = DEQHybridModel(V, d_model=dm, n_patterns=np_, beta=0.5,
                        attract_every=2, error_threshold=0.5,
                        hebbian_lr=0.01).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for ep in range(3):
    model.train()
    x = torch.randint(0, V, (4, 16), device=device)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    ppl = torch.exp(loss).item()
    print(f"  ep{ep}: loss={loss.item():.3f} ppl={ppl:.1f} att={model.get_att_rate():.2f}")

print("DEQHybridModel test PASSED")
