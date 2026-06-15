"""从零训小模型：4K词表 + chunk预测（一次32个token）。
编码器不再是冻住的，和chunk head一起从零学。
"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'
DM, VOCAB = 256, 4096
N_LAYERS = 4
CTX, TGT = 64, 16  # smaller for fast proof-of-concept

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
print(f'DEVICE={DEVICE} DM={DM} V={VOCAB} L={N_LAYERS} CTX={CTX} TGT={TGT}')

# ── Simple WKV-like model (from scratch) ──
class TimeMix(nn.Module):
    def __init__(self):
        super().__init__()
        C = DM
        self.time_mix_k = nn.Parameter(torch.ones(C))
        self.time_mix_v = nn.Parameter(torch.ones(C))
        self.time_mix_r = nn.Parameter(torch.ones(C))
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        for m in [self.key, self.value, self.receptance, self.output]:
            nn.init.xavier_uniform_(m.weight, 0.5)

    def forward(self, x):
        B, T, C = x.shape
        mk, mv, mr = torch.sigmoid(self.time_mix_k), torch.sigmoid(self.time_mix_v), torch.sigmoid(self.time_mix_r)
        xk = x * mk + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mk)
        xv = x * mv + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mv)
        xr = x * mr + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mr)
        k, v, r = self.key(xk), self.value(xv), torch.sigmoid(self.receptance(xr))
        out = []
        h = k.new_zeros(B, C)
        w = k.new_zeros(B, C)
        for t_ in range(T):
            decay = torch.sigmoid(k[:, t_] * 0.1 + 0.5)
            h = decay * h + k[:, t_] * v[:, t_]
            w = decay * w + 1
            out.append(r[:, t_] * (h / (w + 1e-8)))
        return torch.stack(out, dim=1)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM)
        self.ln2 = nn.LayerNorm(DM)
        self.wkv = TimeMix()
        self.ffn = nn.Sequential(
            nn.Linear(DM, DM * 4, bias=False),
            nn.GELU(),
            nn.Linear(DM * 4, DM, bias=False),
        )
    def forward(self, x):
        x = x + self.wkv(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(DM)
        self.chunk_head = nn.Sequential(
            nn.Linear(DM, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        self.pos_embed = nn.Parameter(torch.randn(TGT, DM) * 0.02)
        self.chunk_logits = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        h = self.ln_out(h)
        return h

    def predict_chunk(self, last_h):
        h = self.chunk_head(last_h)
        h = h.unsqueeze(1) + self.pos_embed.unsqueeze(0)
        return self.chunk_logits(h)

model = Model().to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total/1e6:.2f}M')
print(f'  core: {(total - model.embed.weight.numel() - model.chunk_logits.weight.numel())/1e6:.2f}M')
print(f'  chunk_head+logits: {(sum(p.numel() for n,p in model.named_parameters() if "chunk" in n))/1e6:.2f}M')

# ── Data ──
data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]
from collections import Counter
samples = data[np.random.default_rng(42).integers(0, N, 50000)]
top_vocab = [t for t, _ in Counter(samples.tolist()).most_common(VOCAB)]
token_to_id = {t: i for i, t in enumerate(top_vocab)}
OOV = VOCAB - 1
print(f'Vocab coverage: {sum(Counter(samples.tolist())[t] for t in top_vocab)/50000*100:.1f}%')

def get_batch(bsz):
    pos = np.random.randint(0, N - CTX - TGT, (bsz,))
    raw = np.array([data[p:p+CTX+TGT].copy() for p in pos])
    mapped = np.array([[token_to_id.get(int(t), OOV) for t in row] for row in raw], dtype=np.int64)
    x = torch.from_numpy(mapped[:, :CTX]).to(DEVICE)
    y = torch.from_numpy(mapped[:, CTX:CTX+TGT]).to(DEVICE)
    return x, y

LR, BSZ, N_STEPS = 3e-4, 16, 15000
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

model.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    x, y = get_batch(BSZ)
    h = model(x)  # [B, CTX, DM]
    logits = model.predict_chunk(h[:, -1])  # [B, TGT, V]
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    
    # Diversity: entropy of mean distribution across TGT positions
    probs = F.softmax(logits, dim=-1)  # [B, TGT, V]
    mean_dist = probs.mean(dim=1)  # [B, V]
    entropy = -(mean_dist * torch.log(mean_dist + 1e-8)).sum(dim=-1)
    target = math.log(VOCAB)  # ~8.3 for uniform
    div_penalty = F.relu(target - entropy.mean())  # 0=diverse, ~8.3=collapsed
    loss = ce + div_penalty * 0.5
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()

    if step % 1000 == 0:
        with torch.no_grad():
            # Single-token baseline
            l_single = model.chunk_logits(model.chunk_head(h[:, -1]))
            ce_single = F.cross_entropy(l_single, y[:, 0]).item()
            
            ppl = torch.exp(ce).item()
            preds = logits.argmax(-1)
            unique = preds.unique().size(0)
            collapse = unique <= 2

        model.train()
        pbar.set_postfix(
            ce=f'{ce.item():.2f}', div=f'{div_penalty:.2f}',
            ppl=f'{ppl:.0f}',
            s_ce=f'{ce_single:.2f}',
            unique=f'{unique}/{TGT}',
            coll=f'{"Y" if collapse else "N"}',
        )
        torch.save({'model': model.state_dict(), 'step': step},
                   os.path.join(CKPT_DIR, f'chunk_scratch_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
print(f'Final: CE={ce.item():.2f} PPL={torch.exp(ce).item():.0f}')
print(f'Unique tokens: {unique}/{TGT} {"✅" if unique > 2 else "❌ collapse"}')
