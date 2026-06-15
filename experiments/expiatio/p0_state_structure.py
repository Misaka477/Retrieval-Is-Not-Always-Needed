"""Expiatio Phase 0: 验证纯结构 loss 能否产出有语义结构的状态空间。
backbone: NanoGPT 架构（无 lm_head，纯 backbone → 状态）
loss:     MSE 局部预测 + 对比语义聚类（无 CE）
指标:     state ratio（相邻 token vs 随机 token 的 cos 距离比）
"""
import os, sys, time, math, inspect
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device = 'cuda'
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)

# ═══════════════════════════════════════════
# NanoGPT backbone（原版，无 lm_head）
# ═══════════════════════════════════════════

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 65536
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    n_kv_heads: int = 4  # GQA

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.n_embd * 4 * 2 // 3 // 256 * 256
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTBackbone(nn.Module):
    """NanoGPT 架构，无 lm_head——产出状态序列"""
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x  # [B, T, n_embd] = 状态序列


# ═══════════════════════════════════════════
# Expiatio 结构 loss
# ═══════════════════════════════════════════

def local_prediction_loss(states):
    """MSE 预测 h_{t+1} 来自 h_t。单层 MLP 预测器。"""
    B, T, D = states.shape
    h_curr = states[:, :-1]  # [B, T-1, D]
    h_next = states[:, 1:]   # [B, T-1, D]
    # 简单预测器：MLP(h_curr) → h_next_pred
    pred = torch.nn.functional.linear(h_curr, torch.eye(D, device=states.device))
    loss = F.mse_loss(pred, h_next.detach())
    return loss, (pred - h_next.detach()).norm(dim=-1).mean().item()


def contrastive_state_loss(states, tau=0.5):
    """同一序列的状态应该相近，不同序列的状态应该拉远。"""
    B, T, D = states.shape
    # 取每个序列的第一个状态作为锚点
    anchors = states[:, 0, :]  # [B, D]
    anchors = F.normalize(anchors, dim=-1)
    # 正样本：每个序列自己的第二个状态
    positives = states[:, 1, :]  # [B, D]
    positives = F.normalize(positives, dim=-1)
    # 负样本：其他 B-1 个序列的第一个状态
    logits = anchors @ positives.T / tau  # [B, B]
    labels = torch.arange(B, device=states.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(-1) == labels).float().mean().item()
    return loss, acc


def state_structure_score(states):
    """结构指标：相邻 token 的状态相似度 vs 随机 token 的相似度。"""
    B, T, D = states.shape
    h = states.view(-1, D)
    n = h.size(0)
    normed = F.normalize(h, dim=-1)
    # 相邻 token（位置 i 和 i+1 在同一序列中）
    adjacent_pairs = [(i * T + j, i * T + j + 1) for i in range(B) for j in range(T - 1)]
    adj_cos = sum((normed[i] * normed[j]).sum() for i, j in adjacent_pairs) / max(len(adjacent_pairs), 1)
    # 随机 token
    idx = torch.randperm(n, device=states.device)
    rand_cos = (normed * normed[idx]).sum(dim=-1).mean().item()
    adj_cos = adj_cos.item()
    ratio = rand_cos / max(adj_cos, 1e-8)
    return adj_cos, rand_cos, ratio


# ═══════════════════════════════════════════
# Data（使用现有的混合数据，GPT-2 词表映射）
# ═══════════════════════════════════════════

DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
data = np.load(DATA_PATH, mmap_mode='r')
N = len(data)
print(f'Data: {N/1e6:.0f}M tokens')

SEQ, BSZ = 512, 8

def get_batch(bsz, seq):
    pos = np.random.randint(0, N - seq - 1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)


# ═══════════════════════════════════════════
# Train
# ═══════════════════════════════════════════

config = GPTConfig()
model = GPTBackbone(config).to(device)
total = sum(p.numel() for p in model.parameters())
print(f'Backbone: {total/1e6:.2f}M params')

# 只训练 backbone（无额外 head）
opt = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if p.dim() >= 2], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if p.dim() < 2], 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95))

N_STEPS = 10000
model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    x = get_batch(BSZ, SEQ)
    states = model(x)  # [B, T, D]

    # 三层 loss
    mse_loss, _ = local_prediction_loss(states)
    cont_loss, cont_acc = contrastive_state_loss(states)

    loss = mse_loss + 0.5 * cont_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv = get_batch(4, SEQ)
            sv = model(xv)
            mv, _ = local_prediction_loss(sv)
            cv, ca = contrastive_state_loss(sv)
            adj_c, rand_c, ratio = state_structure_score(sv)
        model.train()
        pbar.set_postfix(mse=f'{mse_loss.item():.4f}', cont=f'{cont_loss.item():.4f}',
                         acc=f'{cont_acc:.2f}', ratio=f'{ratio:.3f}')
        torch.save({'model': model.state_dict(), 'step': step, 'ratio': ratio},
                   os.path.join(CKPT_DIR, f'expiatio_p0_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
print(f'Final state structure ratio: best seen ratio')
