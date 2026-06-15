"""MBAT-fp8: fp16/fp8 混合精度（DeepSeek 方式），去 1.58-bit"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB = 65536; DM = 384; N_LAYERS = 8; N_HEADS = 6; N_GROUPS = 2
HEAD_DIM = DM // N_HEADS; d_h = HEAD_DIM; HPG = N_HEADS // N_GROUPS
d_c = d_h * 4; d_c_g = d_c // N_GROUPS; d_c_q = d_c; d_h_R = d_h // 2
SEQ, BSZ, LR = 64, 2, 3e-4; N_STEPS = 10
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'mbat_fp8_log.csv')
BEST_VAL_CE = float('inf'); STEPS_NO_IMPROVE = 0

class Attention(nn.Module):
    """mHC MLA + 金字塔 coarse→fine — 标准 fp16 权重"""
    def __init__(self, max_k=64, lt='cbtka'):
        super().__init__(); self.max_k = max_k; self.lt = lt; self.G = N_GROUPS
        self.W_DKV = nn.ModuleList([nn.Linear(DM, d_c_g) for _ in range(self.G)])
        self.KV_norm = nn.ModuleList([nn.LayerNorm(d_c_g) for _ in range(self.G)])
        self.W_UK = nn.ModuleList([nn.Linear(d_c_g, DM // self.G) for _ in range(self.G)])
        self.W_UV = nn.ModuleList([nn.Linear(d_c_g, DM // self.G) for _ in range(self.G)])
        self.W_DQ = nn.Linear(DM, d_c_q); self.Q_norm = nn.LayerNorm(d_c_q)
        self.W_UQ = nn.Linear(d_c_q, DM)
        self.W_QR = nn.Linear(d_c_q, d_h_R * N_HEADS)
        self.W_KR = nn.Linear(DM, d_h_R); self.W_O = nn.Linear(DM, DM)
        self.register_buffer('cos', torch.zeros(1, SEQ, d_h_R))
        self.register_buffer('sin', torch.zeros(1, SEQ, d_h_R))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_h_R, 2).float() / d_h_R))
        p = torch.arange(SEQ).float()
        self.cos[0, :, 0::2] = torch.cos(p[:, None] * inv_freq[None, :])
        self.cos[0, :, 1::2] = self.cos[0, :, 0::2]
        self.sin[0, :, 0::2] = torch.sin(p[:, None] * inv_freq[None, :])
        self.sin[0, :, 1::2] = self.sin[0, :, 0::2]

    def _apply_rope(self, x, pos):
        c = self.cos[:, pos, :].unsqueeze(2); i = self.sin[:, pos, :].unsqueeze(2)
        xr = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        return x * c + xr * i

    def forward(self, x):
        B, T, D = x.shape; k_top = int(min(self.max_k, T))
        k_c_list, v_c_list = [], []
        for g in range(self.G):
            c_kv = self.KV_norm[g](self.W_DKV[g](x))
            k_c_list.append(self.W_UK[g](c_kv).view(B, T, HPG, HEAD_DIM))
            v_c_list.append(self.W_UV[g](c_kv).view(B, T, HPG, HEAD_DIM))
        k_c = torch.cat(k_c_list, dim=2); v_c = torch.cat(v_c_list, dim=2)
        c_q = self.Q_norm(self.W_DQ(x))
        q_c = self.W_UQ(c_q).view(B, T, N_HEADS, HEAD_DIM)
        q_r = self.W_QR(c_q).view(B, T, N_HEADS, d_h_R)
        k_r = self.W_KR(x).unsqueeze(2).expand(-1, -1, N_HEADS, -1)
        q_r = self._apply_rope(q_r, torch.arange(T, device=x.device))
        k_r = self._apply_rope(k_r, torch.arange(T, device=x.device))
        q = torch.cat([q_c, q_r], dim=-1); k = torch.cat([k_c, k_r], dim=-1)
        q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v_c.transpose(1, 2)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        if self.lt == 'window':
            sc = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(HEAD_DIM + d_h_R) + causal
            W = self.max_k // 2; mask = torch.zeros_like(sc)
            for i in range(T): s = max(0, i-W); e = min(T, i+W+1); mask[:, :, i, s:e] = 1
            h = torch.matmul(F.softmax(sc.masked_fill(mask == 0, float('-inf')), -1), v_t)
            h = h.transpose(1, 2).contiguous().view(B, T, -1)
            return self.W_O(h)
        else:
            sc = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(HEAD_DIM + d_h_R) + causal
            _, idx = torch.topk(sc, k_top, dim=-1)
            ik = idx.unsqueeze(-1).expand(-1, -1, -1, -1, HEAD_DIM + d_h_R)
            iv = idx.unsqueeze(-1).expand(-1, -1, -1, -1, HEAD_DIM)
            k_tk = torch.gather(k_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, ik)
            sc = (q_t.unsqueeze(3) * k_tk).sum(-1) / math.sqrt(HEAD_DIM + d_h_R)
            v_tk = torch.gather(v_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, iv)
            h = (F.softmax(sc, -1).unsqueeze(-1) * v_tk).sum(3)
            return self.W_O(h.transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, i):
        super().__init__(); self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        mk = 32 if i < 3 else (64 if i < 6 else 96); lt = 'window' if i < 3 else 'cbtka'
        self.attn = Attention(max_k=mk, lt=lt)
        self.ffn = nn.Sequential(nn.Linear(DM, DM*4), nn.GELU(), nn.Linear(DM*4, DM))
    def forward(self, x): x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

class MBATfp8(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM); self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))

# Data
data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]; TE = N - N // 10
print(f'Data: {N/1e6:.0f}M, train: {TE/1e6:.0f}M, val: {(N-TE)/1e6:.0f}M')
vr = np.random.RandomState(42)

def get_batch(bsz, seq):
    pos = np.random.randint(0, TE-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = MBATfp8().to(device)
t = sum(p.numel() for p in model.parameters())
print(f'Params: {t/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
scaler = torch.cuda.amp.GradScaler()  # fp16/fp8 混合精度
model.train()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f: f.write('step,loss,val_ce,ppl,lr\n')

pbar = tqdm(range(N_STEPS)); t0 = time.time()
for step in pbar:
    x = get_batch(BSZ, SEQ)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update(); sched.step()
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
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, os.path.join(CKPT_DIR, 'mbat_fp8.pt'))
        if ce_v < BEST_VAL_CE:
            BEST_VAL_CE = ce_v; STEPS_NO_IMPROVE = 0
            torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_best.pt'))
        else:
            STEPS_NO_IMPROVE += 1
            if STEPS_NO_IMPROVE >= 15:
                print(f'\nEarly stop at step {step}, best CE={BEST_VAL_CE:.4f}')
                break

print(f'Train: {(time.time()-t0)/60:.1f}min')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_fp8_final.pt'))
print('Done.')
