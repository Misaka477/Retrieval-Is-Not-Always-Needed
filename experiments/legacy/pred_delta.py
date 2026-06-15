"""Predict delta = h_{t+1} - h_t instead of h_{t+1} directly.
Delta has lower variance → easier to predict.
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ['CUDA_HOME'] = '/home/aquama/miniconda3/envs/natalia'
os.environ['CPATH'] = '/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
os.environ['LD_LIBRARY_PATH'] = '/home/aquama/miniconda3/envs/natalia/lib'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 768

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

data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]

def bb(x):
    pad = (16 - x.size(1) % 16) % 16
    if pad:
        xp = torch.cat([x, torch.zeros(1, pad, dtype=torch.long, device=DEVICE)], 1)
        l, h = bk(xp, return_h=True)
        return l[:, :x.size(1)], h[:, :x.size(1)]
    return bk(x, return_h=True)

# ═══════════════════════════════════════════
# Generate AR trajectory dataset (with deltas)
# ═══════════════════════════════════════════

print('\n=== Generating AR trajectories ===')
N_SEEDS, TRAJ_LEN = 2000, 32
all_h, all_emb, all_delta = [], [], []

t0 = time.time()
for _ in tqdm(range(N_SEEDS)):
    s = torch.randint(0, N - TRAJ_LEN - 1, (1,)).item()
    seed = torch.from_numpy(data[s:s+8].copy()).long().unsqueeze(0).to(DEVICE)
    g = seed.clone()
    h_prev = None

    for pos in range(TRAJ_LEN):
        l, h = bb(g)
        h_curr = h[:, -1]

        if h_prev is not None:
            x_next_id = data[s + 8 + pos].item()
            emb_next = bk.emb(torch.tensor([[x_next_id]], device=DEVICE))[:, 0]
            all_h.append(h_prev.cpu())
            all_emb.append(emb_next.cpu())
            all_delta.append((h_curr - h_prev).cpu())

        h_prev = h_curr

        probs = torch.softmax(l[:, -1].float() / 0.8, -1)
        probs[0, 0] = 0
        nxt = torch.multinomial(probs, 1)
        g = torch.cat([g, nxt], 1)

train_h = torch.cat(all_h, dim=0)
train_emb = torch.cat(all_emb, dim=0)
train_delta = torch.cat(all_delta, dim=0)
N_SAMPLES = train_h.size(0)
print(f'Generated {N_SAMPLES} samples in {(time.time()-t0)/60:.1f}min')
print(f'  h norm: {train_h.norm(dim=-1).mean().item():.2f}')
print(f'  delta norm: {train_delta.norm(dim=-1).mean().item():.2f}')
print(f'  delta norm std: {train_delta.norm(dim=-1).std().item():.2f}')

# ═══════════════════════════════════════════
# Delta Predictor: predict delta from (h_t, emb_{t+1})
# ═══════════════════════════════════════════

class DeltaPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DM * 2, DM), nn.GELU(),
            nn.Linear(DM, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.5)
    def forward(self, h, emb):
        return self.net(torch.cat([h, emb], -1))

pred = DeltaPredictor().to(DEVICE)
print(f'DeltaPredictor: {sum(p.numel() for p in pred.parameters())/1e3:.1f}K')

# ═══════════════════════════════════════════
# Train
# ═══════════════════════════════════════════

LR, EPOCHS = 1e-4, 20
opt = torch.optim.AdamW(pred.parameters(), lr=LR)
pred.train()

print('\n=== Training delta predictor ===')
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    perm = torch.randperm(N_SAMPLES)
    losses = []
    for start in range(0, N_SAMPLES, 1024):
        idx = perm[start:start+1024]
        h = train_h[idx].to(DEVICE)
        emb = train_emb[idx].to(DEVICE)
        target_delta = train_delta[idx].to(DEVICE)

        pred_delta = pred(h, emb)
        loss = F.mse_loss(pred_delta, target_delta)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        hi = train_h[:512].to(DEVICE)
        ei = train_emb[:512].to(DEVICE)
        td = train_delta[:512].to(DEVICE)
        pd = pred(hi, ei)
        tmse = F.mse_loss(pd, td).item()
        tcos = F.cosine_similarity(pd, td).mean().item()
        err = (pd - td).norm(dim=-1)
    pbar.set_postfix(mse=f'{np.mean(losses):.4f}', tmse=f'{tmse:.4f}',
                     tcos=f'{tcos:.3f}', err=f'{err.mean().item():.2f}')

torch.save({'pred': pred.state_dict()}, os.path.join(CKPT_DIR, 'delta_pred.pt'))

# ═══════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════

print('\n=== Analysis ===')
pred.eval()
with torch.no_grad():
    hi = train_h[:1024].to(DEVICE)
    ei = train_emb[:1024].to(DEVICE)
    td = train_delta[:1024].to(DEVICE)
    pd = pred(hi, ei)

    mse = F.mse_loss(pd, td).item()
    cos = F.cosine_similarity(pd, td).mean().item()
    err = (pd - td).norm(dim=-1)

    # Also evaluate via h_pred = h + pred_delta
    h_pred = hi + pd
    h_target = hi + td
    h_mse = F.mse_loss(h_pred, h_target).item()
    h_cos = F.cosine_similarity(h_pred, h_target).mean().item()

    print(f'  Delta: MSE={mse:.4f}  Cos={cos:.3f}')
    print(f'  State (via h+delta): MSE={h_mse:.4f}  Cos={h_cos:.3f}')
    print(f'  Delta error: mean={err.mean().item():.2f} std={err.std().item():.2f}')
    print(f'  Delta err quantiles: 50%={err.quantile(0.5).item():.2f} '
          f'90%={err.quantile(0.9).item():.2f} 95%={err.quantile(0.95).item():.2f}')
    print(f'  Delta norm: (actual) {td.norm(dim=-1).mean().item():.2f} '
          f'(pred) {pd.norm(dim=-1).mean().item():.2f}')

    # Compare with state predictor (previous experiment)
    try:
        from pred_ar import Predictor as StatePred
        ckpt = torch.load(os.path.join(CKPT_DIR, 'pred_ar.pt'), weights_only=False)
        sp = StatePred().to(DEVICE)
        sp.load_state_dict(ckpt['pred'])
        sp.eval()
        sp_out = sp(hi, ei)
        sp_mse = F.mse_loss(sp_out, h_target).item()
        sp_cos = F.cosine_similarity(sp_out, h_target).mean().item()
        print(f'\n  State predictor (pred_ar): MSE={sp_mse:.4f} Cos={sp_cos:.3f}')
        print(f'  Delta predictor improvement: '
              f'{(sp_mse-h_mse)/sp_mse*100:.1f}% MSE, '
              f'{(h_cos-sp_cos)/sp_cos*100:.1f}% Cos')
    except Exception as e:
        print(f'\n  (no state predictor to compare: {e})')

print('\nDone.')
