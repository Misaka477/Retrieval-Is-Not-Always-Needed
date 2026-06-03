"""Train SSM backbone + diffusion objective: denoising CE."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_ssm import MoHESSM

device = 'cuda'
VOCAB, DM, NP = 65536, 256, 384
N_EXP = 4
BSZ, SEQ = 8, 256
LR = 3e-4; N_STEPS = 10000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'train_ssm_diff.csv')

model = MoHESSM(VOCAB, DM, NP, n_experts=N_EXP, topk=2).to(device)
print(f'MoHE-SSM + diffusion: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

with open(CSV_PATH, 'w', newline='') as f:
    f.write('step,loss,acc,mask_rate,gn\n')

model.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    mask_rate = 0.1 + 0.8 * torch.rand(1).item()
    mask = torch.rand(BSZ, SEQ, device=device) < mask_rate
    rand_tokens = torch.randint(10, VOCAB, (BSZ, SEQ), device=device)
    x_corrupt = torch.where(mask, rand_tokens, x)

    logits = model(x_corrupt)
    if isinstance(logits, tuple): logits = logits[0]

    loss = F.cross_entropy(logits.view(-1, VOCAB), x.view(-1))

    with torch.no_grad():
        acc = (logits.argmax(-1) == x).float().mean().item()

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
    opt.step()
    scheduler.step()

    if bi % 250 == 0:
        torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(loss=f'{loss.item():.2f}', acc=f'{acc:.3f}', mr=f'{mask_rate:.2f}', gn=f'{gn:.1f}')
    if bi % 500 == 0 or bi == N_STEPS - 1:
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{acc:.4f},{mask_rate:.4f},{gn:.1f}\n')

torch.save({'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'ssm_diff_final.pt'))
print(f'\nDone - ssm_diff_final.pt')

# ==== Generation comparison ====
print(f'\n{"="*50}\nGeneration test\n{"="*50}')
model.eval()

prompt = torch.randint(10, VOCAB, (1, 4), device=device)

# Greedy
print('\n--- Greedy (24 tokens) ---')
with torch.no_grad():
    gen = prompt.clone()
    for _ in range(20):
        logits = model(gen)
        if isinstance(logits, tuple): logits = logits[0]
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=1)
    out = gen[0].tolist()
    print(f'  Output: {out}')
    uni = len(set(out))
    print(f'  Unique: {uni}/24')
    print(f'  {"*** DIVERSE ***" if uni > 5 else "*** COLLAPSED ***"}')

# Diffusion
def diff_gen(model, steps=50, n_tokens=32):
    with torch.no_grad():
        x = torch.randint(10, VOCAB, (1, n_tokens), device=device)
        for step in range(steps):
            temp = max(0.05, 1.0 - 0.95 * step / steps)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            p = torch.softmax(logits / temp, dim=-1)
            max_p = p.max(dim=-1).values
            thresh = 0.3 + 0.6 * step / steps
            keep = max_p > thresh
            noise = torch.multinomial(p.view(-1, VOCAB), 1).view(1, n_tokens)
            x = torch.where(keep, x, noise)
        return x[0].tolist()

print('\n--- Diffusion 50 steps (32 tokens) ---')
out = diff_gen(model)
print(f'  Output: {out}')
print(f'  Unique: {len(set(out))}/32')

# SSM decay stats
with torch.no_grad():
    x_sample = torch.randint(10, VOCAB, (4, 256), device=device)
    emb = model.embed_norm(model.embed(x_sample))
    dt = F.softplus(model.dt_linear(emb))
    w = torch.exp(-dt)
    print(f'\n  dt: mean={dt.mean().item():.4f} [{dt.min().item():.4f}, {dt.max().item():.4f}]')
    print(f'  w:  mean={w.mean().item():.4f}  [{w.min().item():.4f}, {w.max().item():.4f}]')

print('\nDone.')
