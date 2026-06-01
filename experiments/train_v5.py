"""MoHE-RWKV V5 — masked reconstruction + CE joint training."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV_V5

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 8, 512
LR = 3e-4; N_STEPS = 3000; SAVE_EVERY = 500
RECON_BETA = 0.5
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
OUT_PATH = os.path.join(CKPT_DIR, 'mohe_v5_test.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print("Creating V5 model with masked reconstruction...")
model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, topk=2, recon_beta=RECON_BETA).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)
perm = torch.randperm(nb)

model.train()
pbar = tqdm(range(N_STEPS))
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
    if bi % 200 == 0:
        lr = 3e-5 + (2e-4 - 3e-5) * min(1.0, bi / 2000) if bi < 2000 else 2e-4
        for g in opt.param_groups:
            g['lr'] = lr
        torch.cuda.empty_cache()
    pbar.set_postfix(loss=f'{loss.item():.2f}', recon=f'{loss_recon:.4f}', gn=f'{gn:.1f}')

torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, OUT_PATH)
print(f'Done -> {OUT_PATH}')
