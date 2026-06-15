"""World Model on 12-layer RWKV backbone (official).

Two-phase experiment:
1. Freeze backbone, add small prediction head on top of backbone states.
   Train with prediction loss (MSE + optional contrastive) — NO CE.
2. Train a lightweight language head to translate states → tokens.

Key hypothesis: With 12 layers of WKV producing rich states,
a prediction-based loss can organize the state space semantically
without the collapse seen in 1-2 layer experiments.
"""
import os, sys, time, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Auto-detect CUDA_HOME from conda env
import subprocess
try:
    _result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True, timeout=5)
    if _result.returncode == 0:
        _cuda_home = os.path.dirname(os.path.dirname(_result.stdout.strip()))
        os.environ['CUDA_HOME'] = _cuda_home
        _target = os.path.join(_cuda_home, 'targets', 'x86_64-linux', 'include')
        if os.path.exists(_target):
            os.environ.setdefault('CPATH', '')
            os.environ['CPATH'] = _target + ':' + os.environ['CPATH']
except Exception:
    pass

DEVICE = 'cuda'
DM = 768
VOCAB = 65536
SEQ_LEN = 64
BSZ = 8
LR = 1e-4
N_STEPS = 10000
N_HEAD_STEPS = 5000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
MODEL_PATH = os.path.join(BASE_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth')
DATA_PATH = os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy')

print(f'12L World Model: DM={DM} VOCAB={VOCAB} SEQ={SEQ_LEN} BS={BSZ}')

# ═══════════════════════════════════════════════════
# Load official 12L backbone
# ═══════════════════════════════════════════════════

sys.path.insert(0, os.path.join(BASE_DIR, 'rina'))
from rwkv_v7_demo import RWKV, args

sd = torch.load(MODEL_PATH, map_location='cpu')
for k, v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
        sd[k] = v.float()

backbone = RWKV(args).to(DEVICE)
backbone.load_state_dict(sd, strict=False)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)
print(f'Backbone: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M')

# ═══════════════════════════════════════════════════
# Prediction Head (world model loss on backbone states)
# ═══════════════════════════════════════════════════

class PredictionHead(nn.Module):
    """Predicts next observation in embedding space from backbone state.
    Uses the frozen backbone's own embedding layer as the target space."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(DM, DM),
            nn.GELU(),
            nn.Linear(DM, DM),
        )
        self.predict = nn.Linear(DM, DM)
        for m in [self.proj, self.predict]:
            for p in m.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, 0.1)

    def forward(self, states):
        return self.predict(self.proj(states))


class LanguageHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(DM, VOCAB, bias=False)
        nn.init.xavier_uniform_(self.head.weight, 0.1)

    def forward(self, states):
        return self.head(states)


# ═══════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════

data = np.load(DATA_PATH, mmap_mode='r')
N_TOKENS = data.shape[0]
print(f'Data: {N_TOKENS/1e6:.0f}M tokens')

def get_batch(bsz, seq_len):
    pos = np.random.randint(0, N_TOKENS - seq_len - 1)
    raw = data[pos:pos + bsz * seq_len].copy()
    return torch.from_numpy(raw).long().view(bsz, seq_len).to(DEVICE)

# ═══════════════════════════════════════════════════
# Phase 1: Train prediction head (no CE)
# ═══════════════════════════════════════════════════

print('\n=== Phase 1: Prediction Head (MSE on embedding space) ===')
pred_head = PredictionHead().to(DEVICE)
print(f'Head params: {sum(p.numel() for p in pred_head.parameters())/1e3:.1f}K')

opt = torch.optim.AdamW(pred_head.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

pred_head.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    x = get_batch(BSZ, SEQ_LEN)

    with torch.no_grad():
        logits, states = backbone(x, return_h=True)  # [B, T, DM]

    # Predict next observation's embedding from current state
    pred_next = pred_head(states[:, :-1])  # [B, T-1, DM]
    emb_current = backbone.emb(x[:, 1:])  # [B, T-1, DM]

    mse = F.mse_loss(pred_next, emb_current.detach())

    # Optional: cosine similarity to keep direction
    cos = 1 - F.cosine_similarity(pred_next.view(-1, DM),
                                   emb_current.view(-1, DM)).mean()

    loss = mse + 0.1 * cos

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pred_head.parameters(), 5.0)
    opt.step()
    sched.step()

    if step % 500 == 0:
        pred_head.eval()
        with torch.no_grad():
            tx = get_batch(4, SEQ_LEN)
            tl, ts = backbone(tx, return_h=True)
            tp = pred_head(ts[:, :-1])
            te = backbone.emb(tx[:, 1:])
            tmse = F.mse_loss(tp, te.detach())
            tcos = F.cosine_similarity(tp.view(-1, DM), te.view(-1, DM)).mean()

            # State structure: are backbone states already structured?
            hf = ts[:, :-1].reshape(-1, DM)
            nrm = F.normalize(hf, dim=-1)
            idx = torch.randperm(hf.size(0), device=DEVICE)
            sim_s = (nrm @ nrm.T).mean().item()
            sim_r = (nrm @ nrm[idx].T).mean().item()
        pred_head.train()
        pbar.set_postfix(
            mse=f'{mse.item():.4f}', cos_sim=f'{(1-cos.item()):.3f}',
            tmse=f'{tmse.item():.4f}', tcos=f'{tcos.item():.3f}',
            sr=f'{sim_s/max(sim_r,1e-8):.2f}',
        )
        torch.save({
            'pred_head': pred_head.state_dict(), 'opt': opt.state_dict(),
            'sched': sched.state_dict(), 'step': step,
        }, os.path.join(CKPT_DIR, f'wm12_pred_{step}.pt'))

print(f'\nPhase 1 done in {(time.time()-t0)/60:.1f}min.')

# ═══════════════════════════════════════════════════
# Phase 2: Train language head (read state → token)
# ═══════════════════════════════════════════════════

print('\n=== Phase 2: Language Head ===')
backbone.eval()
pred_head.eval()
lm = LanguageHead().to(DEVICE)
print(f'Head: {sum(p.numel() for p in lm.parameters())/1e6:.2f}M')

opt_lm = torch.optim.AdamW(lm.parameters(), lr=1e-3)
pbar = tqdm(range(N_HEAD_STEPS))

for step in pbar:
    x = get_batch(BSZ, SEQ_LEN)
    with torch.no_grad():
        logits, states = backbone(x, return_h=True)
    l = lm(states[:, :-1])
    ce = F.cross_entropy(l.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt_lm.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(lm.parameters(), 5.0)
    opt_lm.step()
    if step % 1000 == 0:
        pbar.set_postfix(ce=f'{ce.item():.2f}', ppl=f'{torch.exp(ce).item():.0f}')

print(f'\nPhase 2 final: CE={ce.item():.3f} PPL={torch.exp(ce).item():.1f}')

# ═══════════════════════════════════════════════════
# Final analysis
# ═══════════════════════════════════════════════════
print('\n=== Analysis ===')
backbone.eval(); pred_head.eval(); lm.eval()
with torch.no_grad():
    tx = get_batch(16, SEQ_LEN)
    tl, ts = backbone(tx, return_h=True)

    # Prediction accuracy
    pe = backbone.emb(tx[:, 1:])
    pp = pred_head(ts[:, :-1])
    pmse = F.mse_loss(pp, pe).item()
    pcos = F.cosine_similarity(pp.view(-1, DM), pe.view(-1, DM)).mean().item()

    # Language head
    lh = lm(ts[:, :-1])
    lce = F.cross_entropy(lh.reshape(-1, VOCAB), tx[:, 1:].reshape(-1))
    lppl = torch.exp(lce).item()

    # State structure (backbone states)
    hf = ts[:, :-1].reshape(-1, DM)
    nrm = F.normalize(hf, dim=-1)
    idx = torch.randperm(hf.size(0), device=DEVICE)
    sim_s = (nrm @ nrm.T).mean().item()
    sim_r = (nrm @ nrm[idx].T).mean().item()

    print(f'  Prediction: MSE={pmse:.4f} cos={pcos:.3f}')
    print(f'  Language head: CE={lce.item():.3f} PPL={lppl:.1f}')
    print(f'  Backbone state structure: self-cos={sim_s:.4f} shuf-cos={sim_r:.4f} ratio={sim_s/max(sim_r,1e-8):.2f}')
    print(f'  State norm: {hf.norm(dim=-1).mean().item():.2f}')
    print(f'  Dim variance: {hf.var(dim=0).mean().item():.4f}')

    # Per-position vs random structure
    print(f'\n  Backbone states: position-sorted vs shuffled:')
    pos_sorted = F.cosine_similarity(ts[:, :-1].reshape(-1, DM),
                                      ts.new_zeros(ts.size(0)*(SEQ_LEN-1), DM)).mean().item()
    print(f'    mean cos to zero: {pos_sorted:.4f}')
    
    # Train a linear probe on backbone states
    # to see if position can be predicted
    print(f'  Positional structure (state at t vs state at t+k):')
    for gap in [0, 1, 2, 8, 16, 32]:
        if gap < ts.size(1) - 1:
            sim = F.cosine_similarity(ts[:, 0], ts[:, gap]).mean().item()
            print(f'    gap={gap:>2d}: cos = {sim:.4f}')

torch.save({
    'pred_head': pred_head.state_dict(),
    'language_head': lm.state_dict(),
}, os.path.join(CKPT_DIR, 'wm12_final.pt'))
print('\nSaved.')
