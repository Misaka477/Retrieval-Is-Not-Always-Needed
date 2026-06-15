"""Multi-token prediction: predict 32 tokens at once, not 1.
Context=480, target=32. CE on all 32 positions → loss ~32× = can't cheat by repeating.
"""
import os, sys, time, math, glob
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ['CUDA_HOME'] = '/home/aquama/miniconda3/envs/natalia'
os.environ['CPATH'] = '/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
os.environ['LD_LIBRARY_PATH'] = '/home/aquama/miniconda3/envs/natalia/lib'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 768; VOCAB = 65536
CTX, TGT = 480, 32  # context + target lengths

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

class ChunkHead(nn.Module):
    """Project hidden state → logits for TGT tokens simultaneously."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(DM, DM), nn.GELU(),
            nn.Linear(DM, DM),
        )
        self.pos_embed = nn.Parameter(torch.randn(TGT, DM) * 0.02)
        self.head = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, h):
        h = self.proj(h).unsqueeze(1) + self.pos_embed.unsqueeze(0)
        return self.head(h)

head = ChunkHead().to(DEVICE)
print(f'ChunkHead: {sum(p.numel() for p in head.parameters())/1e6:.2f}M')

# ── Data ──
data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N_SEQ = data.shape[0]

# ── Curriculum stages ──
stages = [(480, 32), (448, 64), (384, 128), (256, 256)]
STAGE = 0  # start with first stage

BSZ, LR, N_STEPS = 8, 3e-4, 5000
opt = torch.optim.AdamW(head.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

head.train()
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    ctx, tgt = stages[STAGE]
    pos = np.random.randint(0, N_SEQ - ctx - tgt - 1)
    x = torch.from_numpy(data[pos:pos+ctx+tgt].copy()).long().unsqueeze(0).to(DEVICE)
    x_ctx, x_tgt = x[:, :ctx], x[:, ctx:ctx+tgt]

    with torch.no_grad():
        pad = (16 - ctx % 16) % 16
        xp = torch.cat([x_ctx, torch.zeros(1, pad, dtype=torch.long, device=DEVICE)], 1)
        _, h = bk(xp, return_h=True)
        h = h[:, ctx-1]  # last context position's hidden state

    logits = head(h)  # [1, tgt, V]
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), x_tgt.reshape(-1))

    opt.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
    opt.step(); sched.step()

    if step % 500 == 0:
        with torch.no_grad():
            # Single-token prediction for comparison (from same context)
            l_single, h_s = bk(x_ctx, return_h=True)
            ce_single = F.cross_entropy(
                l_single[:, ctx-1], x_tgt[:, 0]).item()
            
            ppl = torch.exp(ce).item()
            ppl_single = torch.exp(torch.tensor(ce_single)).item()
            
            # Check for collapse: are all predictions the same token?
            preds = logits.argmax(-1)
            unique = preds.unique().size(0)
            collapse = unique <= 1

        head.train()
        pbar.set_postfix(
            ctx=f'{ctx}', tgt=f'{tgt}',
            ce=f'{ce.item():.2f}', ppl=f'{ppl:.0f}',
            single_ce=f'{ce_single:.2f}', single_ppl=f'{ppl_single:.0f}',
            unique=f'{unique}', coll=f'{"Y" if collapse else "N"}',
        )
        
        torch.save({'head': head.state_dict(), 'step': step, 'stage': STAGE},
                   os.path.join(CKPT_DIR, f'chunk_ce_{STAGE}_{step}.pt'))

print(f'Done in {(time.time()-t0)/60:.1f}min')
print(f'Stage {STAGE}: ctx={stages[STAGE][0]} tgt={stages[STAGE][1]}')
print(f'Final CE: {ce.item():.2f} PPL: {torch.exp(ce).item():.0f}')
print(f'Unique tokens in prediction: {unique}/{tgt}')
