"""Predictor-Corrector: predict next state from (state, token).
Small predictor → prediction error → corrector only fired on high error.

Predictor:     h_{t+1}_pred = MLP(h_t, emb(x_{t+1}))
Loss:          MSE(h_{t+1}_pred, h_{t+1})  — trained on actual next state

Corrector:     h_corrected = h_t + gate * MLP(h_t)  — only when error > thresh

No CE anywhere. All on frozen 12L RWKV backbone states.
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ['CUDA_HOME'] = '/home/aquama/miniconda3/envs/natalia'
# Toolkit has complete CUDA headers; add its lib dir for libcudart.so.13 at runtime
_TK_DIR = '/home/aquama/miniconda3/envs/natalia'
_TK_INC = os.path.join(_TK_DIR, 'targets', 'x86_64-linux', 'include')
_TK_LIB = os.path.join(_TK_DIR, 'lib')
os.environ['CPATH'] = _TK_INC + ':' + os.environ.get('CPATH', '')
os.environ['LD_LIBRARY_PATH'] = _TK_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'
DM = 768

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
sys.path.insert(0, os.path.join(BASE_DIR, 'rina'))
from rwkv_v7_demo import RWKV, args

sd = torch.load(os.path.join(BASE_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth'), map_location='cpu')
for k,v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32: sd[k] = v.float()
bk = RWKV(args).to(DEVICE)
bk.load_state_dict(sd, strict=False)
bk.eval()
for p in bk.parameters(): p.requires_grad_(False)
print(f'Backbone: {sum(p.numel() for p in bk.parameters())/1e6:.1f}M')

# ═══════════════════════════════════════════
# Predictor: predict next state from (state, token_emb)
# ═══════════════════════════════════════════

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DM * 2, DM), nn.GELU(),
            nn.Linear(DM, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
    def forward(self, h, emb):
        return self.net(torch.cat([h, emb], -1))

pred = Predictor().to(DEVICE)
print(f'Predictor: {sum(p.numel() for p in pred.parameters())/1e3:.1f}K')

# ═══════════════════════════════════════════
# Data
# ═══════════════════════════════════════════

data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]

def get_batch(bsz, seq_len):
    pos = np.random.randint(0, N - bsz * seq_len - 1)
    return torch.from_numpy(data[pos:pos+bsz*seq_len].copy()).long().view(bsz, seq_len).to(DEVICE)

# ═══════════════════════════════════════════
# Train: predict next state (NO CE)
# ═══════════════════════════════════════════

BSZ, SEQ, LR, STEPS = 8, 64, 1e-4, 5000
opt = torch.optim.AdamW(pred.parameters(), lr=LR)
pred.train()

print('\n=== Training Predictor (predict h_{t+1} from h_t + emb_{t+1}) ===')
pbar = tqdm(range(STEPS))
for step in pbar:
    x = get_batch(BSZ, SEQ)
    with torch.no_grad():
        _, h = bk(x, return_h=True)  # [B, T, DM]
        emb = bk.emb(x)  # [B, T, DM]

    # Use h_t + emb_{t+1} → predict h_{t+1}
    h_pred = pred(h[:, :-1], bk.emb(x[:, 1:]))  # [B, T-1, DM]
    h_target = h[:, 1:]  # [B, T-1, DM]

    mse = F.mse_loss(h_pred, h_target.detach())
    cos = 1 - F.cosine_similarity(h_pred.reshape(-1, DM), h_target.reshape(-1, DM)).mean()
    loss = mse + 0.1 * cos

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
    opt.step()

    if step % 1000 == 0:
        with torch.no_grad():
            tx = get_batch(4, SEQ)
            _, th = bk(tx, return_h=True)
            pred_h = pred(th[:, :-1], bk.emb(tx[:, 1:]))
            target_h = th[:, 1:]
            tmse = F.mse_loss(pred_h, target_h).item()
            tcos = F.cosine_similarity(pred_h.reshape(-1, DM), target_h.reshape(-1, DM)).mean().item()
            # Prediction error distribution
            err = (pred_h - target_h).norm(dim=-1)
        pbar.set_postfix(mse=f'{mse.item():.4f}', cos=f'{(1-cos.item()):.3f}',
                         tmse=f'{tmse:.4f}', tcos=f'{tcos:.3f}',
                         err=f'{err.mean().item():.2f}', err95=f'{err.quantile(0.95).item():.2f}')

# ═══════════════════════════════════════════
# Analyze: prediction error structure
# ═══════════════════════════════════════════

print('\n=== Analysis ===')
pred.eval()
with torch.no_grad():
    tx = get_batch(64, SEQ)
    _, th = bk(tx, return_h=True)
    pred_h = pred(th[:, :-1], bk.emb(tx[:, 1:]))
    target_h = th[:, 1:]

    mse = F.mse_loss(pred_h, target_h).item()
    cos = F.cosine_similarity(pred_h.reshape(-1, DM), target_h.reshape(-1, DM)).mean().item()
    err = (pred_h - target_h).norm(dim=-1)

    print(f'  Predict MSE: {mse:.4f}  Cos: {cos:.3f}')
    print(f'  Error: mean={err.mean().item():.2f} std={err.std().item():.2f}')
    print(f'  Error quantiles: 50%={err.quantile(0.5).item():.2f} '
          f'90%={err.quantile(0.9).item():.2f} 95%={err.quantile(0.95).item():.2f}')
    print(f'  Error > 1.0: {(err > 1.0).float().mean().item()*100:.1f}% of steps')
    print(f'  Error > 0.5: {(err > 0.5).float().mean().item()*100:.1f}% of steps')

    # Compare backbone state structure
    hf = th[:, :-1].reshape(-1, DM)
    nrm = F.normalize(hf, dim=-1)
    idx = torch.randperm(hf.size(0), device=DEVICE)
    sim_s = (nrm @ nrm.T).mean().item()
    sim_r = (nrm @ nrm[idx].T).mean().item()
    print(f'  Backbone state structure ratio: {sim_s/max(sim_r,1e-8):.2f}')

torch.save({'pred': pred.state_dict()}, os.path.join(CKPT_DIR, 'pred_correct_final.pt'))
print('\nSaved predictor.')

# ═══════════════════════════════════════════
# Phase 2: Train Corrector (tiny MLP, only on high-error steps)
# ═══════════════════════════════════════════

print('\n=== Phase 2: Corrector ===')
ERR_THRESH = 150.0  # only train on steps where prediction error > threshold

class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DM, DM // 4), nn.GELU(),
            nn.Linear(DM // 4, DM),
        )
        self.gate = nn.Linear(DM, DM)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.1)

    def forward(self, h):
        return torch.tanh(self.gate(h)) * self.net(h)

corr = Corrector().to(DEVICE)
print(f'Corrector: {sum(p.numel() for p in corr.parameters())/1e3:.1f}K')

opt_c = torch.optim.AdamW(corr.parameters(), lr=1e-4)
pred.eval()
corr.train()

N_STEPS_C = 3000
pbar = tqdm(range(N_STEPS_C))
high_count = 0
total_count = 0

for step in pbar:
    x = get_batch(BSZ, SEQ)
    with torch.no_grad():
        _, h = bk(x, return_h=True)
        h_pred = pred(h[:, :-1], bk.emb(x[:, 1:]))
        h_target = h[:, 1:]
        err = (h_pred - h_target).norm(dim=-1)  # [B, T-1]

    # Only train on high-error steps
    mask = err > ERR_THRESH
    if not mask.any():
        # No high-error steps in this batch, skip
        pbar.update(1)
        pbar.set_postfix(high='0', pct='0%')
        continue

    # Gather high-error samples
    h_high = h[:, :-1][mask]
    h_targ = h_target[mask]

    correction = corr(h_high)
    h_corrected = h_high + correction
    loss_c = F.mse_loss(h_corrected, h_targ)

    opt_c.zero_grad()
    loss_c.backward()
    torch.nn.utils.clip_grad_norm_(corr.parameters(), 5.0)
    opt_c.step()

    high_count = mask.sum().item()
    total_count = mask.numel()

    if step % 500 == 0:
        pbar.set_postfix(loss=f'{loss_c.item():.4f}',
                         high=f'{high_count}', pct=f'{high_count/total_count*100:.1f}%')

print(f'\n  High-error steps: {high_count}/{total_count} ({high_count/total_count*100:.1f}%)')

# ═══════════════════════════════════════════
# Final analysis with corrector
# ═══════════════════════════════════════════

print('\n=== Analysis with Corrector ===')
pred.eval()
corr.eval()
with torch.no_grad():
    tx = get_batch(64, SEQ)
    _, th = bk(tx, return_h=True)
    h_pred = pred(th[:, :-1], bk.emb(tx[:, 1:]))
    h_target = th[:, 1:]

    # Without corrector
    err_raw = (h_pred - h_target).norm(dim=-1)
    mse_raw = F.mse_loss(h_pred, h_target).item()

    # With corrector
    correction = corr(th[:, :-1])
    h_corrected = th[:, :-1] + correction
    err_corr = (h_corrected - h_target).norm(dim=-1)
    mse_corr = F.mse_loss(h_corrected, h_target).item()

    print(f'  No corrector: MSE={mse_raw:.4f}  mean_err={err_raw.mean().item():.2f}')
    print(f'  With corrector: MSE={mse_corr:.4f}  mean_err={err_corr.mean().item():.2f}')
    print(f'  Improvement: {(mse_raw-mse_corr)/mse_raw*100:.1f}%')

    # Error distribution change
    print(f'  High-error (>150) without: {(err_raw > 150).float().mean().item()*100:.1f}%')
    print(f'  High-error (>150) with:    {(err_corr > 150).float().mean().item()*100:.1f}%')

torch.save({'pred': pred.state_dict(), 'corr': corr.state_dict()},
           os.path.join(CKPT_DIR, 'pred_correct_final.pt'))
print('\nDone.')
