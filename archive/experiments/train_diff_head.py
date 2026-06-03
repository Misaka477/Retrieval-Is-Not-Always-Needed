"""Freeze backbone, train only head on diffusion-style corrupted inputs.
Then test diffusion inference vs greedy autoregressive."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model import MoHERWKV
from rina.sample import sample

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 8, 512
LR = 1e-4
N_STEPS = 5000
LR = 5e-4
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_PATH = 'checkpoints/mohe_transferred_latest.pt'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'diff_head_train.csv')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f'Loading backbone from {CKPT_PATH}...')
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
s = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(s, strict=False)
print(f'Loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M backbone')

for p in model.parameters():
    p.requires_grad_(False)
# Don't set wkv_no_grad — let autograd handle the frozen params

orig_head = model.head
model.head = nn.Linear(DM, VOCAB, bias=True).to(device)
model.head.weight.data.copy_(model.embed.weight.data)
model.head.bias.data.copy_(orig_head.bias.data)
model.train()

print(f'Trainable head: {sum(p.numel() for p in model.head.parameters())/1e6:.1f}M')
opt = torch.optim.AdamW(model.head.parameters(), lr=LR)

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

    mask_rate = 0.05 + 0.9 * 0.5 * (1 + math.cos(math.pi * bi / N_STEPS))
    mask = torch.rand(BSZ, SEQ, device=device) < mask_rate
    rand_tokens = torch.randint(10, VOCAB, (BSZ, SEQ), device=device)
    x_corrupt = torch.where(mask, rand_tokens, x)

    logits = model(x_corrupt)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))

    with torch.no_grad():
        acc = (logits.argmax(-1) == x).float().mean().item()

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0).item()
    opt.step()

    if bi % 250 == 0:
        torch.cuda.empty_cache()
    if bi % 50 == 0:
        pbar.set_postfix(loss=f'{loss.item():.2f}', acc=f'{acc:.3f}', mr=f'{mask_rate:.2f}')
    if bi % 100 == 0 or bi == N_STEPS - 1:
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{acc:.4f},{mask_rate:.4f},{gn:.1f}\n')

torch.save({'head': model.head.state_dict()}, os.path.join(CKPT_DIR, 'diff_head.pt'))
print(f'\nDone - diff_head.pt')

# ==== Comparison: greedy vs diffusion generation ====
print(f'\n{"="*50}\nGeneration comparison\n{"="*50}\n')
model.eval()
prompt = torch.randint(10, VOCAB, (1, 4), device=device)  # random 4-token prompt

# Greedy autoregressive
print('--- Greedy autoregressive (5 tokens) ---')
with torch.no_grad():
    gen = prompt.clone()
    for _ in range(5):
        logits = model(gen)[0]
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=1)
        tok = next_tok.item()
        print(f'  token={tok} (id)', end='')
print()

# Diffusion inference (replaces greedy)
print('\n--- Diffusion (30 steps, temp annealing) ---')
with torch.no_grad():
    n_tokens = 32
    x = torch.randint(10, VOCAB, (1, n_tokens), device=device)
    T = 30
    for step in range(T):
        temp = max(0.1, 1.0 - 0.9 * step / T)
        logits = model(x)[0]
        p = torch.softmax(logits / temp, dim=-1)
        max_p, _ = p.max(dim=-1)  # [1, n_tokens]
        threshold = 0.3 + 0.5 * step / T
        keep = max_p > threshold
        noise = torch.randint(10, VOCAB, (1, n_tokens), device=device)
        resample = torch.multinomial(p.view(-1, VOCAB), 1).view(1, n_tokens)
        x = torch.where(keep, x, resample)
    tokens_out = x[0, :16].tolist()
    print(f'  First 16 tokens: {tokens_out}')
    uni = len(set(tokens_out))
    print(f'  Unique tokens: {uni}/16 {"(diverse)" if uni > 4 else "(COLLAPSED)"}')

print()
# Compare with original head
print('--- Greedy with ORIGINAL head (quick sanity) ---')
model.head = orig_head
model.head.requires_grad_(False)
with torch.no_grad():
    gen2 = prompt.clone()
    for _ in range(5):
        logits = model(gen2)[0]
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        gen2 = torch.cat([gen2, next_tok], dim=1)
    print(f'  Prompt:      {prompt[0].tolist()}')
    print(f'  Original:    {gen2[0, -5:].tolist()}')
print()
# Restore trained head
model.head.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'diff_head.pt'), map_location='cpu', weights_only=False)['head'])
print('Restored trained head.')
