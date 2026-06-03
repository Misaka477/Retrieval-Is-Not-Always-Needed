"""V5 Phase 3: End-to-end fine-tuning (all params, low LR)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV_V5

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
CLOUD = True  # False=本地8G卡(BSZ=4), True=云32G(BSZ=12)
BSZ, SEQ = (12, 512) if CLOUD else (4, 512)
LR = 1e-4; N_STEPS = 3000
RECON_BETA = 0.1
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, topk=2, recon_beta=RECON_BETA).to(device)
ckpt = torch.load(os.path.join(CKPT_DIR, 'mohe_v5_phase2.pt'), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
print(f'Loaded Phase 2 checkpoint, full fine-tune (all params {sum(p.numel() for p in model.parameters())/1e6:.2f}M)')

opt = torch.optim.AdamW(model.parameters(), lr=LR)

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
    aux_logits = getattr(model, '_aux_logits', None)
    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss_aux = F.cross_entropy(aux_logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1) if aux_logits is not None else 0.0
    loss_recon = getattr(model, '_recon_loss', 0.0)
    prev_tokens = x[:, :-1]
    prev_scores = logits[:, :-1, :].gather(-1, prev_tokens.unsqueeze(-1)).squeeze(-1)
    loss_rep = torch.sigmoid(prev_scores).mean()
    loss = loss_ce + 0.3 * loss_aux + RECON_BETA * loss_recon + 0.2 * loss_rep
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    pbar.set_postfix(ce=f'{loss_ce.item():.1f}', loss_rep=f'{loss_rep.item():.3f}', gn=f'{gn:.0f}')
    if bi % 200 == 0:
        torch.cuda.empty_cache()

torch.save({'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'mohe_v5_phase3.pt'))
print(f'Phase 3 done -> mohe_v5_phase3.pt')
