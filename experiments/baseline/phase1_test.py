"""Phase 1 验证：纯英文 vs 混合数据 — 从零训 6L Transformer"""
import os, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB = 50257; DM = 384; N_LAYERS = 6; N_HEADS = 6; N_KV_HEADS = 3
HEAD_DIM = DM // N_HEADS
SEQ, BSZ, LR = 128, 8, 3e-4
N_STEPS = 2000

class RoPE(nn.Module):
    def __init__(s):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        s.register_buffer('inv', inv)
    def forward(s, x, pos):
        f = pos.float().unsqueeze(-1) @ s.inv.unsqueeze(0)
        f = torch.cat([f, f], dim=-1)
        xr = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        return x * f.cos() + xr * f.sin()

class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.n_heads = N_HEADS; s.n_kv = N_KV_HEADS; s.hd = HEAD_DIM; s.rep = N_HEADS // N_KV_HEADS
        s.q = nn.Linear(DM, DM, bias=False); s.k = nn.Linear(DM, N_KV_HEADS * HEAD_DIM, bias=False)
        s.v = nn.Linear(DM, N_KV_HEADS * HEAD_DIM, bias=False)
        s.o = nn.Linear(DM, DM, bias=False); s.rope = RoPE()
    def forward(s, x):
        B, T, _ = x.shape
        q = s.q(x).view(B, T, s.n_heads, s.hd).transpose(1, 2)
        k = s.k(x).view(B, T, s.n_kv, s.hd).transpose(1, 2)
        v = s.v(x).view(B, T, s.n_kv, s.hd).transpose(1, 2)
        q = s.rope(q, torch.arange(T, device=x.device))
        k = s.rope(k, torch.arange(T, device=x.device))
        k = k.repeat_interleave(s.rep, dim=1); v = v.repeat_interleave(s.rep, dim=1)
        ca = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(s.hd) + ca
        h = torch.matmul(F.softmax(sc, -1), v).transpose(1, 2).contiguous().view(B, T, -1)
        return s.o(h)

class B(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = nn.LayerNorm(DM); s.ln2 = nn.LayerNorm(DM)
        s.attn = Attn(); s.ffn = nn.Sequential(nn.Linear(DM, DM * 4), nn.GELU(), nn.Linear(DM * 4, DM))
    def forward(s, x): x = x + s.attn(s.ln1(x)); x = x + s.ffn(s.ln2(x)); return x

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(VOCAB, DM)
        s.blocks = nn.ModuleList([B() for _ in range(N_LAYERS)])
        s.ln = nn.LayerNorm(DM); s.head = nn.Linear(DM, VOCAB, bias=False)
    def forward(s, x):
        h = s.emb(x)
        for blk in s.blocks:
            h = blk(h)
        return s.head(s.ln(h))

# 混合数据 (mohe_fw_rwkv_1b.npy 映射到 50K 词表)
data_mixed = np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r')
# 纯英文 (GPT-2 tokenizer, 60K tokens)
data_eng = np.load('data/english_gpt2.npy')

def get_batch(d, bsz, seq):
    pos = np.random.randint(0, len(d)-seq-1, (bsz,))
    x = np.array([d[p:p+seq] for p in pos])
    return torch.from_numpy(x).long().to(device)
def get_batch_val(d, bsz, seq):
    pos = np.random.RandomState(42).randint(0, len(d)-seq-1, (bsz,))
    x = np.array([d[p:p+seq] for p in pos])
    return torch.from_numpy(x).long().to(device)

for name, data_source in [('Pure English', data_eng), ('Mixed', data_mixed)]:
    d = np.clip(data_source[:500000], 0, VOCAB-1) if name == 'Mixed' else data_source
    print(f'\n=== {name} ({len(d)} tokens) ===')
    m = M().to(device); p = sum(p.numel() for p in m.parameters()); print(f'Params: {p/1e6:.2f}M')
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
    m.train()
    for step in range(N_STEPS):
        x = get_batch(d, BSZ, SEQ)
        loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                vx = get_batch_val(d, 4, SEQ)
                ce_v = F.cross_entropy(m(vx)[:, :-1].reshape(-1, VOCAB), vx[:, 1:].reshape(-1)).item()
            m.train()
            print(f'  step {step}: loss={loss.item():.2f} val_ce={ce_v:.2f} ppl={math.exp(ce_v):.0f}')
