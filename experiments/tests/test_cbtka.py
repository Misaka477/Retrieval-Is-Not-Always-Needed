"""Content-Based Top-K Attention + 自适应K + 训练验证"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DM, N_HEADS, HEAD_DIM = 128, 4, 32
SEQ = 64

class AdaptiveCBTKA(nn.Module):
    """自适应 K: 看多少位置由每个query自己决定。"""
    def __init__(self, min_accum=0.9, max_k=64):
        super().__init__()
        self.min_accum = min_accum  # cumulative probability threshold
        self.max_k = max_k  # safety cap
    
    def forward(self, q, k, v):
        B, H, T, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        if self.training:
            # 训练: 只用 top-max_k (固定上限，留给模型学习)
            _, idx = torch.topk(scores, self.max_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
            masked = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(masked, dim=-1)
            k_used = torch.full((B, H, T), self.max_k, device=scores.device)
        else:
            # 推理: 自适应K (softmax累加到阈值)
            prob = F.softmax(scores, dim=-1)
            sorted_prob, _ = prob.sort(dim=-1, descending=True)
            cumsum = sorted_prob.cumsum(dim=-1)
            # 每个query需要多少位置达到min_accum
            k_needed = (cumsum < self.min_accum).sum(dim=-1) + 1  # [B, H, T]
            k_needed = k_needed.clamp(1, self.max_k)
            
            # 用得到的K做CBTKA
            k_used = k_needed
            _, idx = torch.topk(scores, k_needed.max().item(), dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
            # 每个query的K不同: mask掉超出范围的
            range_idx = torch.arange(T, device=scores.device).view(1, 1, 1, T)
            per_query_mask = range_idx < k_needed.unsqueeze(-1).float()
            mask = mask * per_query_mask
            masked = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(masked, dim=-1)
        
        return torch.matmul(attn, v), attn, k_used

# 训练一个CBTKA对比标准attention
class CBTKALayer(nn.Module):
    def __init__(self, max_k=16):
        super().__init__()
        self.q = nn.Linear(DM, DM); self.k = nn.Linear(DM, DM); self.v = nn.Linear(DM, DM)
        self.proj = nn.Linear(DM, DM)
        self.attn = AdaptiveCBTKA(min_accum=0.9, max_k=max_k)
    def forward(self, x):
        B,T,D = x.shape
        q = self.q(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        k = self.k(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        v = self.v(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        h, _, k_used = self.attn(q, k, v)
        h = h.transpose(1,2).contiguous().view(B,T,D)
        return self.proj(h), k_used

class StdAttnLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DM, DM); self.k = nn.Linear(DM, DM); self.v = nn.Linear(DM, DM)
        self.proj = nn.Linear(DM, DM)
    def forward(self, x):
        B,T,D = x.shape
        q = self.q(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        k = self.k(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        v = self.v(x).view(B,T,N_HEADS,HEAD_DIM).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(HEAD_DIM)
        h = torch.matmul(F.softmax(scores, dim=-1), v)
        h = h.transpose(1,2).contiguous().view(B,T,D)
        return self.proj(h), None

class CBTKANet(nn.Module):
    def __init__(self, use_cbtka=True):
        super().__init__()
        self.emb = nn.Embedding(500, DM)
        if use_cbtka:
            self.layers = nn.ModuleList([CBTKALayer(max_k=16) for _ in range(4)])
        else:
            self.layers = nn.ModuleList([StdAttnLayer() for _ in range(4)])
        self.head = nn.Linear(DM, 500)
    def forward(self, x):
        h = self.emb(x)
        k_stats = []
        for l in self.layers:
            h, k = l(h)
            k_stats.append(k)
        return self.head(h), k_stats

# 训练
torch.manual_seed(42)
data = torch.randint(0, 500, (10000,), device=device)

results = {}
for name, use_cbtka in [('Standard', False), ('CBTKA+Adaptive', True)]:
    m = CBTKANet(use_cbtka).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    t0 = time.time()
    losses = []
    all_k = []
    
    for step in range(500):
        x = data[step:step+32].unsqueeze(0)
        logits, k_stats = m(x)
        loss = F.cross_entropy(logits.view(-1,500), x.view(-1))
        if use_cbtka and k_stats:
            all_k.append(torch.stack(k_stats).float().mean().item())
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses[-100:])
    avg_k = np.mean(all_k) if all_k else 64
    t = time.time() - t0
    results[name] = (avg_loss, avg_k, t)

for name, (loss, k, t) in results.items():
    print(f'{name:15s} | loss={loss:.4f} | avg_K={k:.1f}/{SEQ} | time={t:.1f}s')

# 可视化
print(f'\nCBTKA平均每个query只看 {results["CBTKA+Adaptive"][1]:.0f}/{SEQ} 个位置')
print(f'效率: {results["Standard"][1]/results["CBTKA+Adaptive"][1]:.0f}x')
