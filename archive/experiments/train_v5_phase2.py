"""V5 Phase 2: Train router + consolidate + head (experts frozen)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV_V5

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
BSZ, SEQ = 4, 512
LR = 3e-4; N_STEPS = 5000
RECON_BETA = 1.0
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, topk=2, recon_beta=RECON_BETA).to(device)
ckpt = torch.load(os.path.join(CKPT_DIR, 'mohe_v5_phase1.pt'), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
print(f'Loaded Phase 1 checkpoint')

# Freeze experts + time mixing (WKV doesn't need grad in Phase 2)
for p in model.experts.parameters():
    p.requires_grad = False
for n, p in model.named_parameters():
    if 'tmix' in n: p.requires_grad = False
trainable = [p for n, p in model.named_parameters() if p.requires_grad]
print(f'Training params: {sum(p.numel() for p in trainable)/1e6:.2f}M')
print(f'  router: {sum(p.numel() for p in model.router.parameters())/1e3:.0f}K')
print(f'  consolidate: {sum(p.numel() for p in model.consolidate.parameters())/1e3:.0f}K')

opt = torch.optim.AdamW(trainable, lr=LR)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)
N_STEPS = min(nb, N_STEPS)
perm = torch.randperm(nb)

model.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    s = perm[bi % nb] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()
    logits = model(x)
    aux_logits = getattr(model, '_aux_logits', None)
    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss_aux = F.cross_entropy(aux_logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1) if aux_logits is not None else 0.0
    loss_recon = getattr(model, '_recon_loss', 0.0)
    loss = loss_ce + 0.3 * loss_aux + RECON_BETA * loss_recon
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
    opt.step()
    if bi % 200 == 0:
        if bi < 200:
            lr_val = 3e-5 + (LR - 3e-5) * bi / 200
            for g in opt.param_groups: g['lr'] = lr_val
        torch.cuda.empty_cache()
    pbar.set_postfix(ce=f'{loss_ce.item():.1f}', recon=f'{loss_recon:.3f}', gn=f'{gn:.0f}')

torch.save({'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'mohe_v5_phase2.pt'))
print(f'Phase 2 done -> mohe_v5_phase2.pt')
