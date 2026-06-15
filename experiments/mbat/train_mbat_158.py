"""MBAT-1.58: 1.58-bit 三元量化 + MLA + CBTKA + 其他效率优化"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB = 65536; DM = 384; N_LAYERS = 8; N_HEADS = 6; N_GROUPS = 2  # mHC: 2组
HEAD_DIM = DM // N_HEADS; d_h = HEAD_DIM; HPG = N_HEADS // N_GROUPS
d_c = d_h * 4; d_c_g = d_c // N_GROUPS  # 每组独享压缩维度
d_c_q = d_c; d_h_R = d_h // 2
SEQ, BSZ, LR = 512, 8, 3e-4
N_STEPS = 200000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'mbat_158_log.csv')
BEST_VAL_CE = float('inf'); STEPS_NO_IMPROVE = 0

# ════════════════════════════════════════════
# 1.58-bit 三元量化线性层 (STE)
# ════════════════════════════════════════════
class BitLinear(nn.Module):
    """1.58-bit 三元权重 {-1,0,+1} × scale + 直通估计器"""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.weight, 0, 0.02)
        # 可学习的通道级缩放因子 (每组 128 维共享一个 scale)
        self.groups = max(1, out_f // 128)
        self.scale = nn.Parameter(torch.ones(self.groups))

    def forward(self, x):
        w = self.weight
        # 分组缩放
        g = self.groups
        w_g = w.view(g, -1)
        scale = w_g.abs().mean(dim=-1, keepdim=True) * self.scale.view(g, 1)
        w_q = torch.clamp(torch.round(w_g / (scale + 1e-8)), -1, 1) * scale
        w_q = w_q.view_as(w)
        # STE: 前向用量化权重，反向传 fp 梯度
        w_eff = w + (w_q - w).detach()
        return F.linear(x, w_eff, self.bias)

# ════════════════════════════════════════════
# 其他效率优化
# ════════════════════════════════════════════

# 8-bit 激活量化 (训练时前向量化，反向 STE)
def quantize_8bit(x):
    """将激活值量化到 8-bit [-127, 127] × scale"""
    scale = x.abs().max(dim=-1, keepdim=True).values / 127
    x_q = torch.clamp(torch.round(x / (scale + 1e-8)), -127, 127) * scale
    return x + (x_q - x).detach()

# 稀疏门控: 只对 router 置信度高的位置做 attention
class SparseRouter(nn.Module):
    """带置信度阈值的稀疏 router"""
    def __init__(self, d_in):
        super().__init__()
        self.router = BitLinear(d_in, 8)
        self.threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, c_kv):
        # c_kv: [B, T, d_c]
        sc = self.router(c_kv)  # [B, T, R]
        mask = F.softmax(sc, dim=-1).max(dim=-1, keepdim=True).values
        return sc, (mask > torch.sigmoid(self.threshold)).float()

# ════════════════════════════════════════════
# MLA with 1.58-bit
# ════════════════════════════════════════════
class BitMLA(nn.Module):
    """mHC-MLA: 多头分组KV压缩 + 1.58-bit + CBTKA"""
    def __init__(self, max_k=64, lt='cbtka'):
        super().__init__(); self.max_k = max_k; self.lt = lt; self.G = N_GROUPS
        # mHC: 每组独立KV压缩
        self.W_DKV = nn.ModuleList([BitLinear(DM, d_c_g) for _ in range(self.G)])
        self.KV_norm = nn.ModuleList([nn.LayerNorm(d_c_g) for _ in range(self.G)])
        self.W_UK = nn.ModuleList([BitLinear(d_c_g, DM // self.G) for _ in range(self.G)])
        self.W_UV = nn.ModuleList([BitLinear(d_c_g, DM // self.G) for _ in range(self.G)])
        # Q 压缩 (不分组)
        self.W_DQ = BitLinear(DM, d_c_q); self.Q_norm = nn.LayerNorm(d_c_q)
        self.W_UQ = BitLinear(d_c_q, DM)
        # 解耦 RoPE
        self.W_QR = BitLinear(d_c_q, d_h_R * N_HEADS)
        self.W_KR = BitLinear(DM, d_h_R)
        # 输出
        self.W_O = BitLinear(DM, DM)
        # Router (输入为所有组 c_kv 的拼接)
        self.router = SparseRouter(d_c_g * self.G)
        # RoPE
        self.register_buffer('cos', torch.zeros(1, SEQ, d_h_R))
        self.register_buffer('sin', torch.zeros(1, SEQ, d_h_R))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_h_R, 2).float() / d_h_R))
        pos = torch.arange(SEQ).float()
        cos = torch.cos(pos[:, None] * inv_freq[None, :])
        sin = torch.sin(pos[:, None] * inv_freq[None, :])
        self.cos[0, :, 0::2] = cos; self.cos[0, :, 1::2] = cos
        self.sin[0, :, 0::2] = sin; self.sin[0, :, 1::2] = sin

    def _apply_rope(self, x, pos):
        cos = self.cos[:, pos, :].unsqueeze(2)
        sin = self.sin[:, pos, :].unsqueeze(2)
        x_rot = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        return x * cos + x_rot * sin

    def forward(self, x):
        B, T, D = x.shape; k_top = int(min(self.max_k, T))
        x = quantize_8bit(x)
        # mHC: 每组独立KV压缩
        k_c_list, v_c_list = [], []
        for g in range(self.G):
            c_kv = self.KV_norm[g](self.W_DKV[g](x))  # [B, T, d_c_g]
            k_c_list.append(self.W_UK[g](c_kv).view(B, T, HPG, HEAD_DIM))
            v_c_list.append(self.W_UV[g](c_kv).view(B, T, HPG, HEAD_DIM))
        k_c = torch.cat(k_c_list, dim=2)  # [B, T, H, D]
        v_c = torch.cat(v_c_list, dim=2)
        # Q 压缩 (不分组)
        c_q = self.Q_norm(self.W_DQ(x))
        q_c = self.W_UQ(c_q).view(B, T, N_HEADS, HEAD_DIM)
        # 解耦 RoPE
        q_r = self.W_QR(c_q).view(B, T, N_HEADS, d_h_R)
        k_r = self.W_KR(x).unsqueeze(2).expand(-1, -1, N_HEADS, -1)
        q_r = self._apply_rope(q_r, torch.arange(T, device=x.device))
        k_r = self._apply_rope(k_r, torch.arange(T, device=x.device))
        q = torch.cat([q_c, q_r], dim=-1); k = torch.cat([k_c, k_r], dim=-1)
        q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v_c.transpose(1, 2)

        if self.lt == 'window':
            sc = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(HEAD_DIM + d_h_R)
            causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            sc = sc + causal; W = self.max_k // 2; m = torch.zeros_like(sc)
            for i in range(T): s = max(0, i-W); e = min(T, i+W+1); m[:, :, i, s:e] = 1
            at = F.softmax(sc.masked_fill(m == 0, float('-inf')), dim=-1)
        else:
            route_scores, _ = self.router(torch.cat(
                [self.KV_norm[g](self.W_DKV[g](x)) for g in range(self.G)], dim=-1))
            sc_r = torch.matmul(route_scores, route_scores.transpose(-2, -1)) / math.sqrt(8)
            causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            sc_r = sc_r + causal
            _, idx = torch.topk(sc_r, k_top, dim=-1)
            idx = idx.unsqueeze(1).unsqueeze(-1)
            idx_k = idx.expand(-1, N_HEADS, -1, -1, HEAD_DIM + d_h_R)
            idx_v = idx.expand(-1, N_HEADS, -1, -1, HEAD_DIM)
            k_tk = torch.gather(k_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, idx_k)
            sc = (q_t.unsqueeze(3) * k_tk).sum(-1) / math.sqrt(HEAD_DIM + d_h_R)
            at = F.softmax(sc, dim=-1)
            v_tk = torch.gather(v_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, idx_v)
            h = (at.unsqueeze(-1) * v_tk).sum(dim=3)
            return self.W_O(h.transpose(1, 2).contiguous().view(B, T, -1))

        h = torch.matmul(at, v_t).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(h)


class BitBlock(nn.Module):
    def __init__(self, i):
        super().__init__(); self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        mk = 32 if i < 3 else (64 if i < 6 else 96); lt = 'window' if i < 3 else 'cbtka'
        self.attn = BitMLA(max_k=mk, lt=lt)
        self.ffn = nn.Sequential(
            BitLinear(DM, DM*4), nn.GELU(), BitLinear(DM*4, DM),
        )
    def forward(self, x): x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

class MBAT158(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([BitBlock(i) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM); self.head = BitLinear(DM, VOCAB)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.LayerNorm, nn.Embedding)): pass
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

model = MBAT158().to(device)
t = sum(p.numel() for p in model.parameters())
t_bits = sum(p.numel() for n, p in model.named_parameters() if isinstance(p, nn.Parameter))
# 估计：BitLinear 的 weight 占 1.58bit * numel + scale 占 16bit
weight_params = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n and isinstance(p, nn.Parameter))
print(f'Params: {t/1e6:.2f}M (理论 1.58-bit: {weight_params * 1.58 / 8 / 1e6:.2f}MB)')
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f: f.write('step,loss,val_ce,ppl,lr\n')

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
        torch.save({'model': model.state_dict(), 'step': step}, os.path.join(CKPT_DIR, 'mbat_158.pt'))
        if ce_v < BEST_VAL_CE:
            BEST_VAL_CE = ce_v; STEPS_NO_IMPROVE = 0; torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_best.pt'))
        else:
            STEPS_NO_IMPROVE += 1
            if STEPS_NO_IMPROVE >= 15:
                print(f'\nEarly stop at step {step}, best CE={BEST_VAL_CE:.4f}'); break

print(f'Train: {(time.time()-t0)/60:.1f}min')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_158_final.pt'))
print('Done.')
