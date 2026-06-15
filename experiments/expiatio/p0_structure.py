"""Expiatio Phase 0: 位置对比结构学习（L1 only）"""
import os, time, math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device = 'cuda'; CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)

@dataclass
class Config:
    block_size: int = 512; vocab_size: int = 65536; n_layer: int = 12
    n_head: int = 8; n_embd: int = 512; dropout: float = 0.0
    bias: bool = False; n_kv_heads: int = 4

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head; self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v; y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.n_embd * 4 * 2 // 3 // 256 * 256
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__(); self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config); self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x): x = x + self.attn(self.ln_1(x)); x = x + self.mlp(self.ln_2(x)); return x

class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, 0.0, 0.02)
    def forward(self, idx):
        B, T = idx.size(); pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h: x = block(x)
        return self.ln_f(x)

# ── 结构 loss ──

def contrastive_loss(states, tau=0.5, gap=None):
    B, T, D = states.shape
    if gap is None: gap = T // 4
    p = torch.randint(0, T - gap - 1, (B,), device=states.device)
    anchors = states[torch.arange(B), p]; positives = states[torch.arange(B), p + 1]
    negatives = states[torch.arange(B), p + gap]
    idx_other = torch.randperm(B, device=states.device); other_seq = states[idx_other, p]
    a = F.normalize(anchors, dim=-1); pos = F.normalize(positives, dim=-1); neg = F.normalize(negatives + other_seq, dim=-1)
    pos_logits = (a * pos).sum(-1) / tau; neg_logits = a @ neg.T / tau
    logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
    labels = torch.zeros(B, dtype=torch.long, device=states.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(-1) == labels).float().mean().item()
        pd = (a - pos).norm(dim=-1).mean().item(); nd = (a - neg).norm(dim=-1).mean().item()
    return loss, acc, pd, nd, gap

def vicreg_loss(states, gamma=0.5):
    h = states.view(-1, states.size(-1))
    std = torch.sqrt(h.var(dim=0) + 1e-8); var_loss = F.relu(gamma - std).mean()
    h_c = h - h.mean(dim=0); cov = (h_c.T @ h_c) / (h.size(0) - 1)
    off_diag = cov[~torch.eye(cov.size(0), dtype=torch.bool, device=states.device)]
    cov_loss = off_diag.pow(2).mean()
    return var_loss + 0.1 * cov_loss

# ── 指标 ──

def structure_score(states):
    B, T, D = states.shape; h = states.view(-1, D); normed = F.normalize(h, dim=-1)
    adj = [(i * T + j, i * T + j + 1) for i in range(B) for j in range(T - 1)]
    adj_c = sum((normed[i] * normed[j]).sum() for i, j in adj) / max(len(adj), 1)
    idx = torch.randperm(h.size(0), device=states.device); rand_c = (normed * normed[idx]).sum(-1).mean().item()
    adj_c = adj_c.item(); return adj_c, rand_c, rand_c / max(adj_c, 1e-8)

# ── 数据 ──

data = np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r'); N = len(data)
SEQ, BSZ = 512, 8

def get_batch():
    pos = np.random.randint(0, N - SEQ - 1, (BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config = Config(); model = Backbone(config).to(device)
print(f'Backbone: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

opt = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if p.dim() >= 2], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if p.dim() < 2], 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95))

N_STEPS = 10000
CSV_PATH = os.path.join(CKPT_DIR, 'p0_log.csv')
with open(CSV_PATH, 'w') as f:
    f.write('step,s_loss,u_loss,total,acc,pd,nd,adj_c,rand_c,ratio,lr,tau,gap\n')

model.train(); pbar = tqdm(range(N_STEPS)); t0 = time.time()
for step in pbar:
    x = get_batch(); states = model(x)
    tau = max(0.2, 0.8 * (1 - step / N_STEPS))
    s_loss, s_acc, pd, nd, gap = contrastive_loss(states, tau=tau)
    u_loss = vicreg_loss(states)
    loss = s_loss + 0.5 * u_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv = get_batch(); sv = model(xv); adj_c, rand_c, ratio = structure_score(sv)
        model.train()
        pbar.set_postfix(ce=f'{s_loss.item():.3f}', ul=f'{u_loss.item():.4f}', acc=f'{s_acc:.2f}', ratio=f'{ratio:.4f}')
        with open(CSV_PATH, 'a') as f:
            f.write(f'{step},{s_loss.item():.4f},{u_loss.item():.6f},{loss.item():.4f},{s_acc:.4f},{pd:.4f},{nd:.4f},{adj_c:.4f},{rand_c:.4f},{ratio:.4f},{opt.param_groups[0]["lr"]:.2e},{tau:.4f},{gap}\n')
        torch.save({'model': model.state_dict(), 'step': step, 'ratio': ratio}, os.path.join(CKPT_DIR, f'p0_struct_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'p0_struct_final.pt'))
