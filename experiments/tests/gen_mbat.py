"""Standalone generation for MBAT checkpoint. Run anywhere, no imports from training script."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, math, json

device = 'cuda'
DM = 384; VOCAB = 65536; N_LAYERS = 8; N_HEADS = 6; HEAD_DIM = DM // N_HEADS; KV_DIM = HEAD_DIM // 2

class MLA_CBTKA(nn.Module):
    def __init__(self, max_k=64, layer_type='cbtka'):
        super().__init__()
        self.max_k = max_k; self.layer_type = layer_type
        self.q = nn.Linear(DM, DM); self.q_norm = nn.LayerNorm(DM)
        self.kv_proj = nn.Linear(DM, KV_DIM * N_HEADS)
        self.kv_norm = nn.LayerNorm(KV_DIM * N_HEADS)
        self.kv_expand = nn.Linear(KV_DIM * N_HEADS, DM)
        self.v_proj = nn.Linear(DM, KV_DIM * N_HEADS)
        self.v_expand = nn.Linear(KV_DIM * N_HEADS, DM)
        self.out = nn.Linear(DM, DM)
        self.register_buffer('pos_emb', torch.zeros(1, 256, DM))
        pos = torch.arange(256).unsqueeze(1)
        dims = torch.arange(HEAD_DIM // 2).unsqueeze(0)
        pe = torch.zeros(1, 256, HEAD_DIM)
        pe[0, :, 0::2] = torch.sin(pos / 10000 ** (2 * dims / HEAD_DIM))
        pe[0, :, 1::2] = torch.cos(pos / 10000 ** (2 * dims / HEAD_DIM))
        self.pos_emb[:, :, :HEAD_DIM] = pe
        self.forward_cache = {}
    
    def forward(self, x):
        B, T, D = x.shape
        q = self.q_norm(self.q(x)).view(B, T, N_HEADS, HEAD_DIM)
        kv_latent = self.kv_norm(self.kv_proj(x))
        k = self.kv_expand(kv_latent).view(B, T, N_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, T, N_HEADS, KV_DIM)
        pe = self.pos_emb[:, :T, :HEAD_DIM]
        q = (q + pe.view(1, T, 1, HEAD_DIM)).transpose(1, 2)
        k = (k + pe.view(1, T, 1, HEAD_DIM)).transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        
        if self.layer_type == 'window':
            W = self.max_k // 2; mask = torch.zeros_like(scores)
            for i in range(T):
                s, e = max(0, i-W), min(T, i+W+1); mask[:, :, i, s:e] = 1
            attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
        else:
            _, idx = torch.topk(scores, self.max_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
            attn = F.softmax(scores.masked_fill(mask == 0, float('-inf')), dim=-1)
        
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(self.v_expand(h))

class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        mk = 32 if i < 3 else (64 if i < 6 else 96)
        lt = 'window' if i < 3 else 'cbtka'
        self.attn = MLA_CBTKA(max_k=mk, layer_type=lt)
        self.ffn = nn.Sequential(nn.Linear(DM, DM*4), nn.GELU(), nn.Linear(DM*4, DM))
    def forward(self, x):
        return x + self.attn(self.ln1(x)), x + self.ffn(self.ln2(x))

class MBAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h, _ = b(h)
        return self.head(self.ln_out(h))

ckpt = torch.load('checkpoints/mbat_final.pt', map_location='cuda')
m = MBAT().to(device); m.load_state_dict(ckpt['model']); m.eval()
print('Model loaded. Generating...')

data = np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r')
for run in range(3):
    pos = np.random.randint(0, data.shape[0]-80)
    x = torch.from_numpy(data[pos:pos+32].copy()).long().unsqueeze(0).to(device)
    for _ in range(60):
        p = torch.softmax(m(x)[:, -1].float() / 0.8, -1); p[0, 0] = 0
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    tok = {i: l.split(' ')[0] for i, l in enumerate(open('checkpoints/rwkv_vocab_v20230424.txt')) if l.strip()}
    text = ''.join(tok.get(int(i), '?') for i in x[0].tolist())
    print(f'\n--- Run {run+1} ---')
    print(text[:300])
    bg = set(); [bg.add((x[0,i].item(), x[0,i+1].item())) for i in range(x.size(1)-1)]
    print(f'Bigrams: {len(bg)}/{x.size(1)-1}')
