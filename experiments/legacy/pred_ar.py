"""Predictor trained on AR trajectories (not random batches).

AR trajectories produce "dirty" states with accumulated drift,
giving the predictor real correction signals to learn.

Key difference from pred_correct.py:
- Training data from AR generation (states drift off-manifold)
- Same predictor architecture (0.3M)
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
# Generate AR trajectory dataset
# ═══════════════════════════════════════════

print('\n=== Generating AR trajectories ===')
N_SEEDS, TRAJ_LEN = 2000, 32
all_h, all_emb, all_h_next = [], [], []

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
            # Training pair: (h_prev, emb_next) → h_curr
            x_next_id = data[s + 8 + pos].item()
            emb_next = bk.emb(torch.tensor([[x_next_id]], device=DEVICE))[:, 0]  # [1, DM]
            all_h.append(h_prev.cpu())
            all_emb.append(emb_next.cpu())
            all_h_next.append(h_curr.cpu())

        h_prev = h_curr

        # AR step
        probs = torch.softmax(l[:, -1].float() / 0.8, -1)
        probs[0, 0] = 0
        nxt = torch.multinomial(probs, 1)
        g = torch.cat([g, nxt], 1)

train_h = torch.cat(all_h, dim=0)       # [N, DM]
train_emb = torch.cat(all_emb, dim=0)   # [N, DM]
train_h_next = torch.cat(all_h_next, dim=0)  # [N, DM]
N_SAMPLES = train_h.size(0)
print(f'Generated {N_SAMPLES} samples in {(time.time()-t0)/60:.1f}min')
print(f'  h norm: {train_h.norm(dim=-1).mean().item():.2f}')
print(f'  delta norm: {(train_h_next - train_h).norm(dim=-1).mean().item():.2f}')

# ═══════════════════════════════════════════
# Predictor
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
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.5)
    def forward(self, h, emb):
        return self.net(torch.cat([h, emb], -1))

pred = Predictor().to(DEVICE)
print(f'Predictor: {sum(p.numel() for p in pred.parameters())/1e3:.1f}K')

# ═══════════════════════════════════════════
# Train
# ═══════════════════════════════════════════

LR, EPOCHS = 1e-4, 20
opt = torch.optim.AdamW(pred.parameters(), lr=LR)
pred.train()

print('\n=== Training on AR trajectories ===')
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    perm = torch.randperm(N_SAMPLES)
    for start in range(0, N_SAMPLES, 1024):
        idx = perm[start:start+1024]
        h = train_h[idx].to(DEVICE)
        emb = train_emb[idx].to(DEVICE)
        target = train_h_next[idx].to(DEVICE)

        pred_h = pred(h, emb)
        loss = F.mse_loss(pred_h, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 5.0)
        opt.step()

    with torch.no_grad():
        h_test = train_h[:512].to(DEVICE)
        emb_test = train_emb[:512].to(DEVICE)
        target_test = train_h_next[:512].to(DEVICE)
        pred_test = pred(h_test, emb_test)
        tmse = F.mse_loss(pred_test, target_test).item()
        tcos = F.cosine_similarity(pred_test, target_test).mean().item()
        err = (pred_test - target_test).norm(dim=-1)
    pbar.set_postfix(mse=f'{loss.item():.4f}', tmse=f'{tmse:.4f}',
                     tcos=f'{tcos:.3f}', err=f'{err.mean().item():.2f}')

torch.save({'pred': pred.state_dict()}, os.path.join(CKPT_DIR, 'pred_ar.pt'))

# ═══════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════

print('\n=== Analysis ===')
pred.eval()
with torch.no_grad():
    h_test = train_h[:1024].to(DEVICE)
    emb_test = train_emb[:1024].to(DEVICE)
    target_test = train_h_next[:1024].to(DEVICE)
    pred_test = pred(h_test, emb_test)

    mse = F.mse_loss(pred_test, target_test).item()
    cos = F.cosine_similarity(pred_test, target_test).mean().item()
    err = (pred_test - target_test).norm(dim=-1)

    print(f'  Predict MSE: {mse:.4f}  Cos: {cos:.3f}')
    print(f'  Error: mean={err.mean().item():.2f} std={err.std().item():.2f}')
    print(f'  Error quantiles: 50%={err.quantile(0.5).item():.2f} '
          f'90%={err.quantile(0.9).item():.2f} 95%={err.quantile(0.95).item():.2f}')

    # Compare with random batch predictor
    # (load the random predictor if available)
    try:
        ckpt = torch.load(os.path.join(CKPT_DIR, 'pred_correct_final.pt'), weights_only=False)
        pred_rand = Predictor().to(DEVICE)
        pred_rand.load_state_dict(ckpt['pred'])
        pred_rand.eval()
        p_rand = pred_rand(h_test, emb_test)
        r_mse = F.mse_loss(p_rand, target_test).item()
        r_cos = F.cosine_similarity(p_rand, target_test).mean().item()
        print(f'\n  Random-batch predictor: MSE={r_mse:.4f} Cos={r_cos:.3f}')
        print(f'  AR predictor improvement: MSE {mse:.4f} vs {r_mse:.4f}')
    except:
        print('\n  (no random-batch predictor to compare)')

print('\nDone.')
