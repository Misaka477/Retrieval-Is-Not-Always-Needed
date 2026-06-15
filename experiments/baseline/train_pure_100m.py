"""Pure Transformer 101M — GQA, RoPE, weight tying, decay groups, scaled init."""
import os, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB = 65536; DM = 512; N_LAYERS = 12; N_HEADS = 8; N_KV_HEADS = 4
HEAD_DIM = DM // N_HEADS
SEQ, BSZ, LR = 512, 16, 3e-4; N_STEPS = 100000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'pure_100m_log.csv')
BEST_VAL_CE = float('inf')

class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x, pos):
        freqs = pos.float().unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        freqs = torch.cat([freqs, freqs], dim=-1)
        xr = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        return x * freqs.cos() + xr * freqs.sin()

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS; self.n_kv_heads = N_KV_HEADS
        self.head_dim = HEAD_DIM; self.n_rep = N_HEADS // N_KV_HEADS
        self.q_proj = nn.Linear(DM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, DM, bias=False)  # c_proj
        self.rope = RoPE()
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(T, device=x.device)
        q, k = self.rope(q, pos), self.rope(k, pos)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + causal
        h = torch.matmul(F.softmax(sc, dim=-1), v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(h)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        self.attn = Attention()
        self.ffn = nn.ModuleDict(dict(
            w1=nn.Linear(DM, DM * 4 * 2 // 3 // 256 * 256, bias=False),  # gate
            w2=nn.Linear(DM * 4 * 2 // 3 // 256 * 256, DM, bias=False),  # down
            w3=nn.Linear(DM, DM * 4 * 2 // 3 // 256 * 256, bias=False),  # up
        ))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        ffn = self.ffn
        x = x + ffn['w2'](F.silu(ffn['w1'](x)) * ffn['w3'](x))
        return x

class PureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init_weights)
        # ③ c_proj 缩放: 所有残差投影初始化缩小
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('ffn.w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYERS))
        # ① weight tying
        self.head.weight = self.emb.weight

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))

data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]; TE = N - N // 10
print(f'Data: {N/1e6:.0f}M, train: {TE/1e6:.0f}M, val: {(N-TE)/1e6:.0f}M')
vr = np.random.RandomState(42)

def get_batch(bsz, seq):
    pos = np.random.randint(0, TE-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = PureTransformer().to(device)
t = sum(p.numel() for p in model.parameters())
print(f'Params: {t/1e6:.2f}M')

# ② 权重衰减分组: 2D 参数 decay, 1D 不 decay
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': nodecay_params, 'weight_decay': 0.0},
], lr=LR, betas=(0.9, 0.95))  # beta2=0.95 matches LLM practice
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train()
with open(CSV_PATH, 'w') as f:
    f.write('step,loss,val_ce,ppl,lr\n')

pbar = tqdm(range(N_STEPS)); t0 = time.time()
for step in pbar:
    x = get_batch(BSZ, SEQ)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            p_ = vr.randint(TE, N-SEQ-1, 8)
            vx = torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in p_]).to(device)
            ce_v = F.cross_entropy(model(vx)[:, :-1].reshape(-1, VOCAB), vx[:, 1:].reshape(-1)).item()
        model.train()
        pbar.set_postfix(loss=f'{loss.item():.2f}', ce_v=f'{ce_v:.2f}', ppl=f'{math.exp(ce_v):.0f}')
        with open(CSV_PATH, 'a') as f:
            f.write(f'{step},{loss.item():.4f},{ce_v:.4f},{math.exp(ce_v):.0f},{sched.get_last_lr()[0]:.2e}\n')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step},
                   os.path.join(CKPT_DIR, 'pure_100m.pt'))
        if ce_v < BEST_VAL_CE:
            BEST_VAL_CE = ce_v
            torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'pure_100m_best.pt'))

print(f'Train: {(time.time()-t0)/60:.1f}min, Best CE: {BEST_VAL_CE:.4f}')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'pure_100m_final.pt'))
