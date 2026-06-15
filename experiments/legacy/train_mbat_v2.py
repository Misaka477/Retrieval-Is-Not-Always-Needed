"""MBAT-v2: CBTKA + DeepSeek V4 MLA + CSA 混合注意力。"""
import os, sys, time, math, types
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

device = 'cuda'
VOCAB = 50257; DM = 512; N_LAYERS = 12; N_HEADS = 8
HEAD_DIM = DM // N_HEADS
KV_DIM = HEAD_DIM // 2  # MLA压缩: KV维度砍半
SEQ, BSZ, LR = 256, 2, 2e-4
N_STEPS = 8000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)

class MLA_CBTKA(nn.Module):
    """DeepSeek V4 MLA + CBTKA: Q全秩, KV压缩, 自适应Top-K"""
    def __init__(self, max_k=64, layer_type='cbtka'):
        super().__init__()
        self.max_k = max_k
        self.layer_type = layer_type  # 'window', 'cbtka', 'hybrid'
        
        # Q全秩投影
        self.q = nn.Linear(DM, DM)
        self.q_norm = nn.LayerNorm(DM)
        
        # MLA式的KV压缩: 投影到低维再解压
        self.kv_proj = nn.Linear(DM, KV_DIM * N_HEADS)
        self.kv_norm = nn.LayerNorm(KV_DIM * N_HEADS)
        self.kv_expand = nn.Linear(KV_DIM * N_HEADS, DM)  # 解压回DM
        
        self.v_proj = nn.Linear(DM, KV_DIM * N_HEADS)  # V同样压缩
        self.v_expand = nn.Linear(KV_DIM * N_HEADS, DM)  # V解压回DM
        
        # RoPE (简化: 位置编码加在QK上)
        self.register_buffer('pos_emb', self._create_pos_emb(SEQ, HEAD_DIM))
        
        self.out = nn.Linear(DM, DM)
    
    def _create_pos_emb(self, T, D):
        pos = torch.arange(T).unsqueeze(1)
        dims = torch.arange(D // 2).unsqueeze(0)
        theta = 10000 ** (-2 * dims / D)
        pe = torch.zeros(1, T, D)
        pe[0, :, 0::2] = torch.sin(pos * theta)
        pe[0, :, 1::2] = torch.cos(pos * theta)
        return pe
    
    def forward(self, x):
        B, T, D = x.shape
        
        # MLA: Q全秩, KV压缩
        q = self.q_norm(self.q(x)).view(B, T, N_HEADS, HEAD_DIM)
        
        # KV压缩 → 潜空间 → 解压 (MLA核心)
        kv_latent = self.kv_norm(self.kv_proj(x))
        k = self.kv_expand(kv_latent).view(B, T, N_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, T, N_HEADS, KV_DIM)
        
        # 简化RoPE
        pe = self.pos_emb[:, :T]
        q = q + pe.view(1, T, 1, HEAD_DIM)
        k = k + pe.view(1, T, 1, HEAD_DIM)
        
        # Attention
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        
        if self.layer_type == 'window':
            # 滑动窗口: 只看前后W个位置
            W = self.max_k // 2
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
            mask = torch.zeros_like(scores)
            for i in range(T):
                start, end = max(0, i-W), min(T, i+W+1)
                mask[:, :, i, start:end] = 1
            attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
            k_used = torch.full((B, N_HEADS, T), min(T, 2*W+1), device=x.device)
        else:
            # CBTKA: 自适应Top-K
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
            
            if self.training:
                _, idx = torch.topk(scores, self.max_k, dim=-1)
                mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
                attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
                k_used = torch.full((B, N_HEADS, T), self.max_k, device=x.device)
            else:
                prob = F.softmax(scores, dim=-1)
                sorted_p, _ = prob.sort(dim=-1, descending=True)
                k_needed = (sorted_p.cumsum(dim=-1) < 0.9).sum(dim=-1) + 1
                k_needed = k_needed.clamp(1, self.max_k)
                _, idx = torch.topk(scores, k_needed.max().item(), dim=-1)
                mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
                rng = torch.arange(T, device=x.device).view(1, 1, 1, T)
                mask = mask * (rng < k_needed.unsqueeze(-1).float())
                attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
                k_used = k_needed
        
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        h = self.v_expand(h)  # 解压到DM
        return self.out(h), k_used

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        # 底层(window) → 中层(cbtka) → 顶层(hybrid cbtka + full)
        if layer_id < 4: lt = 'window'; mk = 32
        elif layer_id < 8: lt = 'cbtka'; mk = 64
        else: lt = 'cbtka'; mk = 128
        self.attn = MLA_CBTKA(max_k=mk, layer_type=lt)
        self.ffn = nn.Sequential(nn.Linear(DM, DM*4), nn.GELU(), nn.Linear(DM*4, DM))
    
    def forward(self, x):
        h, k = self.attn(self.ln1(x))
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, k

class MBATv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, x):
        h = self.emb(x)
        all_k = []
        for l in self.layers:
            h, k = l(h)
            all_k.append(k)
        return self.head(self.ln_out(h)), all_k

# Data
tok = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer'); tok.pad_token = tok.eos_token
vocab_file = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
texts = []
with open(vocab_file, encoding='utf-8') as f:
    for line in f:
        l = line.strip()
        if l and len(l) > 1 and ord(l[0]) >= 32: texts.append(l)
corpus = ' '.join(texts[:100000]) * 30
ids = tok.encode(corpus, max_length=3000000, truncation=True)
data = np.array(ids, dtype=np.int32); N = data.shape[0]; print(f'Data: {N} tokens')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = MBATv2().to(device)
total = sum(p.numel() for p in model.parameters())
emb_sz = model.emb.weight.numel(); head_sz = model.head.weight.numel()
print(f'Params: {total/1e6:.2f}M (emb: {emb_sz/1e6:.2f}M, head: {head_sz/1e6:.2f}M, core: {(total-emb_sz-head_sz)/1e6:.2f}M)')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train(); pbar = tqdm(range(N_STEPS)); t0 = time.time()

for step in pbar:
    x = get_batch(BSZ, SEQ)
    logits, all_k = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    
    if step % 1000 == 0:
        per_layer_k = [k.float().mean().item() for k in all_k]
        avg_k = np.mean(per_layer_k)
        pbar.set_postfix(loss=f'{loss.item():.2f}', k=f'{avg_k:.0f}/{SEQ}',
                         l1=f'{per_layer_k[0]:.0f}', l6=f'{per_layer_k[6]:.0f}', l11=f'{per_layer_k[11]:.0f}')

print(f'Train: {(time.time()-t0)/60:.1f}min')
avg_k_final = np.mean([k.float().mean().item() for k in all_k])
print(f'Final avg_K: {avg_k_final:.0f}/{SEQ} ({avg_k_final/SEQ*100:.0f}%)')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_v2.pt'))
print('Done.')
