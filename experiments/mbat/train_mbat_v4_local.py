"""MBAT-v4: 金字塔 coarse→fine + Q范数路由 + 自适应k + mHC + 1.58-bit"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB = 65536; DM = 384; N_LAYERS = 8; N_HEADS = 6; N_GROUPS = 2
HEAD_DIM = DM // N_HEADS; d_h = HEAD_DIM; HPG = N_HEADS // N_GROUPS
d_c = d_h * 4; d_c_g = d_c // N_GROUPS; d_c_q = d_c; d_h_R = d_h // 2
SEQ, BSZ, LR = 128, 2, 3e-4; N_STEPS = 50000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'mbat_v4_log.csv')
BEST_VAL_CE = float('inf'); STEPS_NO_IMPROVE = 0

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.weight, 0, 0.02)
        self.groups = max(1, out_f // 128)
        self.scale = nn.Parameter(torch.ones(self.groups))

    def forward(self, x):
        w = self.weight; g = self.groups
        w_g = w.view(g, -1)
        scale = w_g.abs().mean(dim=-1, keepdim=True) * self.scale.view(g, 1)
        w_q = torch.clamp(torch.round(w_g / (scale + 1e-8)), -1, 1) * scale
        w_q = w_q.view_as(w)
        w_eff = w + (w_q - w).detach()
        return F.linear(x, w_eff, self.bias)

def quantize_8bit(x):
    scale = x.abs().max(dim=-1, keepdim=True).values / 127
    x_q = torch.clamp(torch.round(x / (scale + 1e-8)), -127, 127) * scale
    return x + (x_q - x).detach()

# ════════════════════════════════════════════
# 金字塔注意力 (coarse→fine) + Q范数路由 + 自适应k
# ════════════════════════════════════════════
class PyramidAttention(nn.Module):
    """三级金字塔: stride=4 coarse → stride=2 mid → top-k exact"""
    def __init__(self, max_k=64, lt='cbtka'):
        super().__init__(); self.max_k = max_k; self.lt = lt; self.G = N_GROUPS
        # mHC KV 投影
        self.W_DKV = nn.ModuleList([BitLinear(DM, d_c_g) for _ in range(self.G)])
        self.KV_norm = nn.ModuleList([nn.LayerNorm(d_c_g) for _ in range(self.G)])
        self.W_UK = nn.ModuleList([BitLinear(d_c_g, DM // self.G) for _ in range(self.G)])
        self.W_UV = nn.ModuleList([BitLinear(d_c_g, DM // self.G) for _ in range(self.G)])
        # Q 压缩
        self.W_DQ = BitLinear(DM, d_c_q); self.Q_norm = nn.LayerNorm(d_c_q)
        self.W_UQ = BitLinear(d_c_q, DM)
        # 解耦 RoPE
        self.W_QR = BitLinear(d_c_q, d_h_R * N_HEADS)
        self.W_KR = BitLinear(DM, d_h_R)
        self.W_O = BitLinear(DM, DM)
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

    def _coarse_attention(self, q_t, k_t, stride, max_k, causal, T):
        """在降采样序列上做 coarse attention"""
        B, H, Tq, D = q_t.shape
        # 降采样
        q_coarse = q_t[:, :, ::stride, :]
        k_coarse = k_t[:, :, ::stride, :]
        Tc = q_coarse.size(2)
        # 在 coarse 序列上算 attention (full or top-k)
        k_top = min(max_k, Tc)
        sc = torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) / math.sqrt(D)
        # coarse-level causal mask
        causal_c = torch.triu(torch.full((Tc, Tc), float('-inf'), device=q_t.device), diagonal=1)
        sc = sc + causal_c
        # 插值回完整分辨率
        scores_full = torch.zeros(B, H, T, T, device=q_t.device)
        scores_full[:, :, ::stride, ::stride] = sc
        scores_full = F.interpolate(scores_full.view(B*H, 1, T, T), size=(T, T), mode='bilinear', align_corners=False).view(B, H, T, T)
        scores_full = scores_full.masked_fill(causal == float('-inf'), float('-inf'))
        # 自适应 k: 基于 Q 范数
        q_norm = q_t.norm(dim=-1)
        thresholds = q_norm.sort(dim=-1).values[:, :, T // 4 * 3] if T >= 4 else q_norm.mean(dim=-1)
        per_query_k = torch.where(q_norm > thresholds.unsqueeze(-1), max_k * 2, max_k)
        return scores_full, per_query_k

    def _pyramid_coarse(self, q_t, k_t, k_top, T):
        """三级金字塔: stride=4 → stride=2 → exact"""
        B, H, Tq, D = q_t.shape
        causal = torch.triu(torch.full((T, T), float('-inf'), device=q_t.device), diagonal=1)
        
        # Level 1: stride=4 coarse (16x cheaper)
        scores_l1, per_q_k = self._coarse_attention(q_t, k_t, 4, k_top * 2, causal, T)
        
        # 根据 Q 范数决定哪些位置需要 level 2
        q_norm = q_t.norm(dim=-1)
        need_mid = q_norm > q_norm.mean(dim=-1, keepdim=True)
        
        # Level 2: stride=2, 只在 need_mid 位置做
        scores_l2 = scores_l1.clone()
        for b in range(B):
            for h in range(H):
                mid_pos = need_mid[b, h].nonzero(as_tuple=True)[0]
                if len(mid_pos) == 0: continue
                # 对 mid_pos 位置，用 stride=2 的 coarse attention
                q_mid = q_t[b:b+1, h:h+1, mid_pos, :]
                # 降采样 key 到 stride=2
                k_mid = k_t[b:b+1, h:h+1, ::2, :]
                sc_mid = torch.matmul(q_mid, k_mid.transpose(-2, -1)) / math.sqrt(D)
                q_idx = mid_pos.float(); k_idx = torch.arange(0, T, 2, device=q_t.device).float()
                sc_mid = sc_mid + causal_mid[:, :k_mid.size(2)]
                # 映射回去
                scores_l2[b:b+1, h:h+1, mid_pos, :] = -1e9
                scores_l2[b:b+1, h:h+1, mid_pos, ::2] = sc_mid
        
        # Level 3: top-k exact from the fused scores
        topk_vals, idx = torch.topk(scores_l2, k_top, dim=-1)
        return idx, topk_vals

    def forward(self, x):
        B, T, D = x.shape; k_top = int(min(self.max_k, T))
        x = quantize_8bit(x)
        # mHC KV 压缩
        k_c_list, v_c_list = [], []
        for g in range(self.G):
            c_kv = self.KV_norm[g](self.W_DKV[g](x))
            k_c_list.append(self.W_UK[g](c_kv).view(B, T, HPG, HEAD_DIM))
            v_c_list.append(self.W_UV[g](c_kv).view(B, T, HPG, HEAD_DIM))
        k_c = torch.cat(k_c_list, dim=2); v_c = torch.cat(v_c_list, dim=2)
        # Q
        c_q = self.Q_norm(self.W_DQ(x))
        q_c = self.W_UQ(c_q).view(B, T, N_HEADS, HEAD_DIM)
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
            h = torch.matmul(at, v_t).transpose(1, 2).contiguous().view(B, T, -1)
            return self.W_O(h)
        else:
            # 小序列直接用精确top-k (金字塔只在长序列有意义)
            if T > 256:
                idx, _ = self._pyramid_coarse(q_t, k_t, k_top, T)
            else:
                sc_full = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(HEAD_DIM + d_h_R)
                causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
                sc_full = sc_full + causal
                _, idx = torch.topk(sc_full, k_top, dim=-1)
            
            # 对 top-k 位置做精确 attention
            idx_k = idx.unsqueeze(-1).expand(-1, -1, -1, -1, HEAD_DIM + d_h_R)
            idx_v = idx.unsqueeze(-1).expand(-1, -1, -1, -1, HEAD_DIM)
            k_tk = torch.gather(k_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, idx_k)
            sc = (q_t.unsqueeze(3) * k_tk).sum(-1) / math.sqrt(HEAD_DIM + d_h_R)
            at = F.softmax(sc, dim=-1)
            v_tk = torch.gather(v_t.unsqueeze(3).expand(-1, -1, -1, k_top, -1), 2, idx_v)
            h = (at.unsqueeze(-1) * v_tk).sum(dim=3)
            return self.W_O(h.transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, i):
        super().__init__(); self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        mk = 32 if i < 3 else (64 if i < 6 else 96)
        lt = 'window' if i < 3 else 'cbtka'
        self.attn = PyramidAttention(max_k=mk, lt=lt)
        self.ffn = nn.Sequential(BitLinear(DM, DM*4), nn.GELU(), BitLinear(DM*4, DM))
    def forward(self, x): x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

class MBATv4(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM); self.head = BitLinear(DM, VOCAB)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.LayerNorm, nn.Embedding)): pass
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

model = MBATv4().to(device)
t = sum(p.numel() for p in model.parameters())
print(f'Params: {t/1e6:.2f}M')
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
        torch.save({'model': model.state_dict(), 'step': step}, os.path.join(CKPT_DIR, 'mbat_v4.pt'))
        if ce_v < BEST_VAL_CE:
            BEST_VAL_CE = ce_v; STEPS_NO_IMPROVE = 0
            torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_best.pt'))
        else:
            STEPS_NO_IMPROVE += 1
            if STEPS_NO_IMPROVE >= 15:
                print(f'\nEarly stop at step {step}, best CE={BEST_VAL_CE:.4f}')
                break

print(f'Train: {(time.time()-t0)/60:.1f}min')
torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'mbat_v4_final.pt'))
print('Done.')
