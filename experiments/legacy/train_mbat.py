"""MBAT: Multi-Block Adaptive Transformer — DeepSeek V4 + CBTKA 融合架构。

设计原则:
1. 底层 (1-4): 滑动窗口 + CBTKA (K=16)，局部模式匹配
2. 中层 (5-8): 纯 CBTKA (K=64)，内容路由
3. 顶层 (9-12): CBTKA (K=128) + 可选的全局 attention，跨段推理

每层的K由模型自适应决定 (90% accumulation threshold)。"""
import os, sys, time, math, types
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

device = 'cuda'
VOCAB, DM = 50257, 384
N_LAYERS, N_HEADS = 6, 6
HEAD_DIM = DM // N_HEADS
SEQ, BSZ, LR = 256, 8, 3e-4
N_STEPS = 5000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)

class CBTKA(nn.Module):
    """Content-Based Top-K Attention with adaptive K and KV compression."""
    def __init__(self, max_k=32, kv_ratio=0.5):
        super().__init__()
        self.max_k = max_k
        self.min_accum = 0.9
        # KV压缩: 类似DeepSeek V4的MLA，但输出维度减半
        self.kv_dim = int(DM * kv_ratio)
        self.q = nn.Linear(DM, self.kv_dim); self.k = nn.Linear(DM, self.kv_dim)
        self.v = nn.Linear(DM, self.kv_dim); self.proj = nn.Linear(self.kv_dim, DM)
        self.q_norm = nn.LayerNorm(self.kv_dim); self.kv_norm = nn.LayerNorm(self.kv_dim)
        
    def forward(self, x, use_full=False):
        B, T, D = x.shape
        q = self.q_norm(self.q(x))
        k = self.kv_norm(self.k(x))
        v = self.v(x)
        q = q.view(B, T, N_HEADS, -1).transpose(1, 2)
        k = k.view(B, T, N_HEADS, -1).transpose(1, 2)
        v = v.view(B, T, N_HEADS, -1).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        if use_full:
            attn = F.softmax(scores, dim=-1)
            k_used = torch.full((B, N_HEADS, T), T, device=x.device)
        else:
            if self.training:
                _, idx = torch.topk(scores, self.max_k, dim=-1)
                mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
                masked = scores.masked_fill(mask == 0, float('-inf'))
                attn = F.softmax(masked, dim=-1)
                k_used = torch.full((B, N_HEADS, T), self.max_k, device=x.device)
            else:
                prob = F.softmax(scores, dim=-1)
                sorted_p, _ = prob.sort(dim=-1, descending=True)
                k_needed = (sorted_p.cumsum(dim=-1) < self.min_accum).sum(dim=-1) + 1
                k_needed = k_needed.clamp(1, self.max_k)
                _, idx = torch.topk(scores, k_needed.max().item(), dim=-1)
                mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
                range_idx = torch.arange(T, device=x.device).view(1, 1, 1, T)
                mask = mask * (range_idx < k_needed.unsqueeze(-1).float())
                attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
                k_used = k_needed
        
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(h), k_used

class MBATLayer(nn.Module):
    def __init__(self, layer_id, max_k):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(DM)
        self.ln2 = nn.LayerNorm(DM)
        # 底层用普通attention+CBTKA混合，顶层用纯CBTKA
        self.attn = CBTKA(max_k=max_k)
        self.ffn = nn.Sequential(
            nn.Linear(DM, DM * 4), nn.GELU(), nn.Linear(DM * 4, DM),
        )
    
    def forward(self, x):
        attn_out, k_used = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, k_used

class MBAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        # 每层不同的max_k，底层低、顶层高
        layer_k = [16, 16, 32, 32, 64, 64]
        self.layers = nn.ModuleList([MBATLayer(i, layer_k[i]) for i in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, x):
        h = self.emb(x)
        all_k = []
        for l in self.layers:
            h, k = l(h)
            all_k.append(k)
        return self.head(self.ln_out(h)), all_k

# ── Data ──
tok = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer')
tok.pad_token = tok.eos_token

# Build corpus from RWKV vocab
import json
vocab_file = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
text_chunks = []
with open(vocab_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line) > 1 and ord(line[0]) >= 32:
            text_chunks.append(line)
corpus = ' '.join(text_chunks[:50000]) * 50
all_ids = tok.encode(corpus, max_length=2000000, truncation=True)
data = np.array(all_ids, dtype=np.int32)
N = data.shape[0]; print(f'Corpus: {N} tokens')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

# ── Train ──
model = MBAT().to(device)
total = sum(p.numel() for p in model.parameters())
print(f'Params: {total/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train(); pbar = tqdm(range(N_STEPS)); t0 = time.time()
k_log = []

for step in pbar:
    x = get_batch(BSZ, SEQ)
    logits, all_k = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        avg_k = torch.stack([k.float().mean() for k in all_k]).mean().item()
        k_log.append(avg_k)
        pbar.set_postfix(loss=f'{loss.item():.2f}', avg_k=f'{avg_k:.0f}/{SEQ}')

print(f'Train: {(time.time()-t0)/60:.1f}min')
print(f'Final avg_K: {k_log[-1] if k_log else "N/A":.0f}/{SEQ}')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat.pt'))
print('Saved.')
