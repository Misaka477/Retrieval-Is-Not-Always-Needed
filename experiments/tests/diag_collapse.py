"""诊断：MBAT 训练 CE 异常。BSZ=2, SEQ=128, 打印每一步的 logits/预测分布"""
import os, sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

device = 'cuda'
VOCAB = 65536; DM = 384; N_LAYERS = 8; N_HEADS = 6
HEAD_DIM = DM // N_HEADS; KV_DIM = HEAD_DIM // 2
SEQ, BSZ, LR = 128, 2, 3e-4

class Attention(nn.Module):
    def __init__(self, max_k=64, lt='cbtka'):
        super().__init__(); self.max_k = max_k; self.lt = lt
        self.q = nn.Linear(DM, DM); self.qn = nn.LayerNorm(DM)
        self.kp = nn.Linear(DM, KV_DIM * N_HEADS); self.kn = nn.LayerNorm(KV_DIM * N_HEADS)
        self.ke = nn.Linear(KV_DIM * N_HEADS, DM)
        self.vp = nn.Linear(DM, KV_DIM * N_HEADS); self.ve = nn.Linear(KV_DIM * N_HEADS, DM)
        self.o = nn.Linear(DM, DM)
        self.register_buffer('pb', torch.zeros(1, SEQ, DM))
        p = torch.arange(SEQ).unsqueeze(1); d = torch.arange(HEAD_DIM // 2).unsqueeze(0)
        pe = torch.zeros(1, SEQ, HEAD_DIM)
        pe[:, :, 0::2] = torch.sin(p / 10000 ** (2 * d / HEAD_DIM)).unsqueeze(0)
        pe[:, :, 1::2] = torch.cos(p / 10000 ** (2 * d / HEAD_DIM)).unsqueeze(0)
        self.pb[:, :, :HEAD_DIM] = pe

    def forward(self, x):
        B, T, D = x.shape; k = min(self.max_k, T)
        q = self.qn(self.q(x)).view(B, T, N_HEADS, HEAD_DIM)
        kl = self.kn(self.kp(x)); ke = self.ke(kl).view(B, T, N_HEADS, HEAD_DIM)
        v = self.vp(x).view(B, T, N_HEADS, KV_DIM)
        pe = self.pb[:, :T, :HEAD_DIM]
        q = (q + pe.view(1, T, 1, HEAD_DIM)).transpose(1, 2)
        ke = (ke + pe.view(1, T, 1, HEAD_DIM)).transpose(1, 2)
        v = v.transpose(1, 2)
        sc = torch.matmul(q, ke.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        if self.lt == 'window':
            W = self.max_k // 2; m = torch.zeros_like(sc)
            for i in range(T): s = max(0, i-W); e = min(T, i+W+1); m[:, :, i, s:e] = 1
            attn = F.softmax(sc.masked_fill(m == 0, float('-inf')), dim=-1)
        else:
            _, idx = torch.topk(sc, k, dim=-1)
            m = torch.zeros_like(sc).scatter_(-1, idx, 1.0)
            attn = F.softmax(sc.masked_fill(m == 0, float('-inf')), dim=-1)
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(self.ve(h))

class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        mk = 16 if i < 3 else 32; lt = 'window' if i < 3 else 'cbtka'
        self.attn = Attention(max_k=mk, lt=lt)
        self.ffn = nn.Sequential(nn.Linear(DM, DM*4), nn.GELU(), nn.Linear(DM*4, DM))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM); self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks: h = blk(h)
        return self.head(self.ln(h))

# Data
data = np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r')
N = data.shape[0]; TE = N - N // 10
print(f'Data: {N/1e6:.0f}M, train: 0-{TE/1e6:.0f}M, val: {TE/1e6:.0f}M-{N/1e6:.0f}M')

m = Model().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
total = sum(p.numel() for p in m.parameters())
print(f'Params: {total/1e6:.2f}M')

# 固定验证集
val_rng = np.random.RandomState(42)
val_pos = val_rng.randint(TE, N-SEQ-1, 8)
val_x = torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in val_pos]).to(device)

m.train()
for step in range(200):
    pos = np.random.randint(0, TE-SEQ-1, BSZ)
    x = torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)
    logits = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
    opt.step()

    if step % 50 == 0:
        m.eval()
        with torch.no_grad():
            # 训练 CE
            ce_train = loss.item()
            # 验证 CE
            lv = m(val_x)
            ce_val = F.cross_entropy(lv[:, :-1].reshape(-1, VOCAB), val_x[:, 1:].reshape(-1)).item()
            # Logits 范围
            print(f'[{step:3d}] train_ce={ce_train:.4f} val_ce={ce_val:.4f} '
                  f'logits=[{lv.min():.1f},{lv.max():.1f}] mean={lv.mean():.1f} '
                  f'std={lv.std():.1f}')
            # 检查预测分布：是否有大量 -inf 或 0
            probs = F.softmax(lv, dim=-1)
            print(f'       max_prob={probs.max():.6f} min_prob={probs.min():.6f}')
            # 预测的 token 是否越来越集中
            preds_flat = lv.argmax(-1).reshape(-1)
            top_freq = preds_flat.mode().values.item()
            top_pct = (preds_flat == top_freq).float().mean().item() * 100
            print(f'       top_pred_token={top_freq} appears={top_pct:.0f}%')
            # 如果出现大量相同预测，进一步排查
            if top_pct > 50:
                # 看 attention 输出是不是全 0
                h_test = m.blocks[0].attn(m.blocks[0].ln1(m.emb(val_x)))
                print(f'       Layer0 attn output: mean={h_test.mean():.4f} std={h_test.std():.4f}')
        m.train()

print('\nDone.')
