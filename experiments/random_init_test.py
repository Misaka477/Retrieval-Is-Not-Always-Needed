"""MoHE-RWKV from scratch - random init test (5000 steps)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 2, 4096
LR = 3e-4; N_STEPS = 5000; SAVE_EVERY = 500
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

print("Creating model with RANDOM init (no weight transfer)...")
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, route_noise=0.0, topk=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
n_val = max(1, len(ids) // 20)
ids_train = ids[:-n_val]
nb = (len(ids_train)-1)//(BSZ*SEQ)
perm = torch.randperm(nb)

model.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    s = perm[bi % nb] * BSZ * SEQ
    x = ids[s:s+BSZ*SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss = loss + getattr(model, '_last_aux_loss', 0.0)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    if bi % 200 == 199:
        lr = 3e-5 + (2e-4 - 3e-5) * min(1.0, (bi - 199) / 2000) if bi < 2199 else 2e-4
        for g in opt.param_groups: g['lr'] = lr
    if bi % 200 == 0: torch.cuda.empty_cache()
    pbar.set_postfix(loss=f'{loss.item():.2f}', gn=f'{gn:.1f}')

torch.save({'step': N_STEPS, 'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'mohe_random_init_test.pt'))
print('Done - saved to mohe_random_init_test.pt')
