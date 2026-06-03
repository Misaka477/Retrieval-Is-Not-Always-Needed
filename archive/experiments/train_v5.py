"""MoHE-RWKV V5 — masked reconstruction + CE, 30K steps."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV_V5

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
BSZ, SEQ = 8, 512
LR = 3e-4; N_STEPS = 30000; SAVE_EVERY = 1500
RECON_BETA = 0.5
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'mohe_v5.csv')
RESUME_CKPT = os.path.join(CKPT_DIR, 'mohe_v5_latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, topk=2, recon_beta=RECON_BETA).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

start_step = 0
if os.path.exists(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step']
    total_loss = ckpt.get('total_loss', 0.0)
    print(f'Resumed from step {start_step}')
else:
    total_loss = 0.0
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,loss,recon,ce,gn,lr\n')

print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)
N_STEPS = min(nb, N_STEPS)
perm = torch.randperm(nb)

model.train()
pbar = tqdm(range(start_step, N_STEPS), initial=start_step, total=N_STEPS)
for bi in pbar:
    s = perm[bi % nb] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()
    logits = model(x)
    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss_recon = getattr(model, '_recon_loss', 0.0)
    loss_aux = getattr(model, '_aux_loss', 0.0)
    loss = loss_ce + RECON_BETA * loss_recon + loss_aux
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    if bi < 2000:
        lr = 3e-5 + (2e-4 - 3e-5) * (bi / 2000) if bi < 2000 else 2e-4
        for g in opt.param_groups: g['lr'] = lr
    else:
        p_decay = (bi - 2000) / (N_STEPS - 2000)
        lr_val = 1e-5 + (2e-4 - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(1, p_decay)))
        for g in opt.param_groups: g['lr'] = lr_val
    total_loss += loss.item()
    if bi % 200 == 0:
        torch.cuda.empty_cache()
    pbar.set_postfix(loss=f'{loss.item():.2f}', recon=f'{loss_recon:.3f}', gn=f'{gn:.0f}')

    if bi % SAVE_EVERY == SAVE_EVERY - 1 or bi == N_STEPS - 1:
        ppl = float(torch.exp(torch.tensor(total_loss / (bi + 1))))
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.2f},{loss_recon:.4f},{loss_ce.item():.2f},{gn:.1f},{opt.param_groups[0]["lr"]:.2e}\n')
        torch.save({'step': bi + 1, 'model': model.state_dict(), 'opt': opt.state_dict(), 'total_loss': total_loss},
                   RESUME_CKPT + '.tmp')
        os.replace(RESUME_CKPT + '.tmp', RESUME_CKPT)
        if (bi + 1) % (SAVE_EVERY * 10) == 0:
            import shutil; shutil.copy(RESUME_CKPT, os.path.join(CKPT_DIR, f'mohe_v5_{bi+1}.pt'))
        torch.cuda.empty_cache()
        print(f'\n  Saved step {bi+1}: loss={loss.item():.2f}, recon={loss_recon:.4f}')

print('Done')
