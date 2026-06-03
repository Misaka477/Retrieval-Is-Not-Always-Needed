"""MoHE-FFN + CE — 109M. Quick validation: does FFN expert avoid collapse at scale?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_ffn import MoHEFFN

device = 'cuda'
VOCAB, DM = 65536, 768
N_EXP = 12
BSZ, SEQ = 8, 512
LR = 2e-4; N_STEPS = 24000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'train_ffn_ce_109m.csv')

model = MoHEFFN(VOCAB, DM, n_experts=N_EXP, topk=2).to(device)
nparam = sum(p.numel() for p in model.parameters())
print(f'MoHE-FFN 109M: {nparam/1e6:.1f}M params')

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

start_step = 0
resume_path = os.path.join(CKPT_DIR, 'ffn_ce_109m_latest.pt')
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step']
    print(f'Resumed from step {start_step}')

with open(CSV_PATH, 'a' if start_step > 0 else 'w', newline='') as f:
    if start_step == 0:
        f.write('step,loss,ppl,gn,lr\n')

model.train()
pbar = tqdm(range(start_step, N_STEPS), initial=start_step, total=N_STEPS)
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    logits = model(x)
    if isinstance(logits, tuple): logits = logits[0]
    loss = F.cross_entropy(logits.view(-1, VOCAB), x.view(-1))

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
    opt.step()
    scheduler.step()

    if bi % 500 == 0: torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(loss=f'{loss.item():.2f}', ppl=f'{math.exp(min(loss.item(),20)):.1f}', gn=f'{gn:.1f}')
    if bi % 1000 == 0 or bi == N_STEPS - 1:
        ppl = math.exp(min(loss.item(), 20))
        lr_cur = opt.param_groups[0]['lr']
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{ppl:.4f},{gn:.1f},{lr_cur:.2e}\n')
        torch.save({'step': bi+1, 'model': model.state_dict(), 'opt': opt.state_dict()},
                   resume_path + '.tmp')
        os.replace(resume_path + '.tmp', resume_path)

torch.save({'step': N_STEPS, 'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'ffn_ce_109m_final.pt'))
print(f'\nDone — ffn_ce_109m_final.pt')

# ==== Generation test ====
print(f'\n{"="*50}\nGeneration test\n{"="*50}')
model.eval()
prompt = torch.randint(10, VOCAB, (1, 4), device=device)

print('\n--- Greedy (24 tokens) ---')
with torch.no_grad():
    gen = prompt.clone()
    for _ in range(20):
        logits = model(gen)
        if isinstance(logits, tuple): logits = logits[0]
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=1)
    out = gen[0].tolist()
    uni = len(set(out))
    print(f'  Unique: {uni}/24')
    print(f'  {"*** DIVERSE ***" if uni > 5 else "*** COLLAPSED ***"}')

print('\nDone.')
