"""Train MoHE-SSM (Mamba-style selective decay) from scratch with CE."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_ssm import MoHESSM

device = 'cuda'
VOCAB, DM, NP = 65536, 256, 384
N_EXP = 4
BSZ, SEQ = 8, 256
LR = 3e-4; N_STEPS = 5000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'train_ssm.csv')

model = MoHESSM(VOCAB, DM, NP, n_experts=N_EXP, topk=2).to(device)
print(f'MoHE-SSM: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

with open(CSV_PATH, 'w', newline='') as f:
    f.write('step,loss,ppl,gn\n')

model.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss = F.cross_entropy(logits.view(-1, VOCAB), x.view(-1))

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
    opt.step()
    scheduler.step()

    if bi % 250 == 0:
        torch.cuda.empty_cache()
    if bi % 200 == 0:
        ppl = math.exp(min(loss.item(), 20))
        pbar.set_postfix(loss=f'{loss.item():.2f}', ppl=f'{ppl:.1f}', gn=f'{gn:.1f}')
    if bi % 500 == 0 or bi == N_STEPS - 1:
        ppl = math.exp(min(loss.item(), 20))
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{ppl:.4f},{gn:.1f}\n')

# Save checkpoint
torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': bi},
           os.path.join(CKPT_DIR, 'ssm_checkpoint.pt'))

# ==== Generation test ====
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
    print(f'  Unique tokens: {len(set(out))}/24')
    if len(set(out)) <= 5:
        print('  *** COLLAPSED (≤5 unique tokens) ***')
    else:
        print('  *** DIVERSE ***')

# Diffusion inference
def diffusion_generate(model, n_tokens=32, steps=30):
    with torch.no_grad():
        x = torch.randint(10, VOCAB, (1, n_tokens), device=device)
        for step in range(steps):
            temp = max(0.05, 1.0 - 0.95 * step / steps)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            p = torch.softmax(logits / temp, dim=-1)
            max_p = p.max(dim=-1).values
            threshold = 0.3 + 0.6 * step / steps
            keep = max_p > threshold
            resample = torch.multinomial(p.view(-1, VOCAB), 1).view(1, n_tokens)
            x = torch.where(keep, x, resample)
        return x[0].tolist()

print('\n--- Diffusion 30 steps (32 tokens) ---')
out = diffusion_generate(model)
print(f'  Output: {out}')
print(f'  Unique tokens: {len(set(out))}/32')

# WKV decay tracking: compute mean dt per batch
with torch.no_grad():
    x_sample = torch.randint(10, VOCAB, (4, 256), device=device)
    emb = model.embed_norm(model.embed(x_sample))
    dt = F.softplus(model.dt_linear(emb))
    print(f'\n  dt stats: mean={dt.mean().item():.4f}, min={dt.min().item():.4f}, max={dt.max().item():.4f}')
    w = torch.exp(-dt)
    print(f'  w stats:  mean={w.mean().item():.4f}, min={w.min().item():.4f}, max={w.max().item():.4f}')
    # Lower w = less decay = longer memory
    print(f'  Effective decay per step: {w.mean().item():.4f}')
    print(f'  Effective context (~5 steps): {int(-5 / math.log(max(w.mean().item(), 1e-6)))} tokens' if w.mean().item() < 0.999 else '  Near-perfect memory')

print('\nDone.')
