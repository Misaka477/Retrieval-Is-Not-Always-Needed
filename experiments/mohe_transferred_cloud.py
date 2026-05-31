"""MoHE-RWKV transferred — clean baseline for 1B tokens."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 5, 4096
LR = 3e-4
CKPT_DIR = 'checkpoints'
RESUME_CKPT = os.path.join(CKPT_DIR, 'mohe_transferred_latest.pt')
INIT_CKPT = os.path.join(CKPT_DIR, 'mohe_transferred_init.pt')
CSV_PATH = os.path.join(CKPT_DIR, 'mohe_transferred_v3.csv')
DATA_PATH = os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy')
SAVE_EVERY = 1500

os.makedirs(CKPT_DIR, exist_ok=True)

model = MoHERWKV(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, route_noise=0.0, topk=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

start_step = 0; total_loss = 0.0

if os.path.exists(RESUME_CKPT):
    print(f'Resuming from {RESUME_CKPT} ...')
    ckpt = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
    sd = ckpt['model']
    for k in list(sd.keys()):
        if k.startswith('prev_route') or '_batch_' in k: del sd[k]
    model.load_state_dict(sd, strict=False)
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step']
    total_loss = ckpt['total_loss']
    print(f'  Resumed at step {start_step}')
else:
    print('No checkpoint found, initializing from transferred weights ...')
    ckpt = torch.load(INIT_CKPT, map_location='cpu', weights_only=False)
    for k in list(ckpt.keys()):
        if k.startswith('prev_route') or '_batch_' in k: del ckpt[k]
    model.load_state_dict(ckpt, strict=False)
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,ppl,loss,lr,grad_norm,aux_loss,route_ent,exp_sim,val_ppl\n')

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,ppl,loss,lr,grad_norm,aux_loss,route_ent,exp_sim,val_ppl\n')

model.train()
ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
n_val = max(1, len(ids) // 20)
ids_train = ids[:-n_val]
ids_val = ids[-n_val:]
print(f'Train: {len(ids_train):,}, Val: {len(ids_val):,}')

@torch.no_grad()
def eval_val():
    model.eval()
    vl = 0.0; vc = 0; vs = 0
    for s in range(0, len(ids_val) - BSZ * SEQ, BSZ * SEQ):
        x = ids_val[s:s+BSZ*SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
        out = model(x); logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
        vl += loss.item() * x.size(0); vc += x.size(0); vs += 1
        if vs >= 100: break
    model.train()
    return float(torch.exp(torch.tensor(vl / max(vc, 1))))

nb = (len(ids_train)-1)//(BSZ*SEQ)
N_STEPS = max(start_step, nb) + nb  # full epoch of new data
perm = torch.randperm(nb)

pbar = tqdm(range(start_step, N_STEPS), initial=start_step, total=N_STEPS)
for bi in pbar:
    s = perm[bi % nb] * BSZ * SEQ
    x = ids[s:s+BSZ*SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss = loss + getattr(model, '_last_aux_loss', 0.0)
    loss.backward()
    model.finish_training_step()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    if torch.isnan(torch.tensor(gn)):
        pbar.write(f'  NaN grad at step {bi+1}, skipping')
        opt.zero_grad(); continue
    opt.step()
    total_loss += loss.item()
    if bi - start_step < 2000:
        t = (bi - start_step) / 2000
        lr = 3e-5 + (2e-4 - 3e-5) * t
    else:
        denom = max(1, N_STEPS - start_step - 2000)
        p = (bi - start_step - 2000) / denom
        lr = 1e-5 + (2e-4 - 1e-5) * 0.5 * (1 + math.cos(math.pi * max(0.0, min(1.0, p))))
    for g in opt.param_groups:
        g['lr'] = lr

    if bi % 500 == 0:
        torch.cuda.empty_cache()

    if bi % SAVE_EVERY == SAVE_EVERY - 1 or bi == N_STEPS - 1:
        ppl = float(torch.exp(torch.tensor(total_loss / (bi + 1))))
        aux = getattr(model, '_last_aux_loss', 0.0)
        ent = getattr(model, '_last_route_entropy', 0.0)
        val_ppl = eval_val()
        with torch.no_grad():
            pts = [e.patterns.data for e in model.experts]
            sims = [(pts[i] @ pts[j].T).mean().item() for i in range(model.n_experts) for j in range(i+1, model.n_experts)]
            exp_sim = max(0.0, sum(sims) / len(sims)) if sims else 0.0
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{ppl:.1f},{loss.item():.2f},{opt.param_groups[0]["lr"]:.2e},{gn:.4f},{aux:.6f},{ent:.4f},{exp_sim:.4f},{val_ppl:.1f}\n')
        torch.save({'step': bi+1, 'model': model.state_dict(), 'opt': opt.state_dict(),
                     'total_loss': total_loss},
                    RESUME_CKPT + '.tmp')
        os.replace(RESUME_CKPT + '.tmp', RESUME_CKPT)
        if (bi+1) % (SAVE_EVERY * 15) == 0:
            import shutil, glob
            shutil.copy(RESUME_CKPT, os.path.join(CKPT_DIR, f'mohe_{bi+1}.pt'))
            snaps = sorted(glob.glob(os.path.join(CKPT_DIR, 'mohe_[0-9]*.pt')))
            for f in snaps[:-2]: os.remove(f)
        torch.cuda.empty_cache()
        print(f'\n  Saved step {bi+1}')
