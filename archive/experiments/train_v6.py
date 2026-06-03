"""Train MoHE-v6: 6× [WKV → MoE(topk=1)]. Untied head."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_v6 import MoHEv6

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 512
N_LAYERS = 6
BSZ, SEQ = 8, 512
LR = 3e-4; N_STEPS = 24000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'train_v6.csv')

model = MoHEv6(VOCAB, DM, NP, n_layers=N_LAYERS).to(device)
nparam = sum(p.numel() for p in model.parameters())
print(f'MoHE-v6: {nparam/1e6:.1f}M ({N_LAYERS}× layers, 2 expert/layer, topk=1)')

# Download & load official RWKV 0.1B weights for initialization
rwkv_ckpt_path = os.path.join(CKPT_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth')
if not os.path.exists(rwkv_ckpt_path):
    print('Downloading RWKV 0.1B from HuggingFace...')
    os.system('pip install -q huggingface_hub 2>/dev/null')
    os.system(f'hf download BlinkDL/rwkv7-g1 rwkv7-g1d-0.1b-20260129-ctx8192.pth --local-dir {CKPT_DIR}')

print('Loading RWKV 0.1B weights...')
rwkv = torch.load(rwkv_ckpt_path, map_location='cpu', weights_only=False)

for our_i in range(N_LAYERS):
    off_i = our_i * 2
    layer = model.layers[our_i]
    b = f'blocks.{off_i}.'
    
    # Official uses w0/w1/w2 for decay, not tmix_w. Skip tmix_w (leave random).
    layer.tmix_r.weight.data.copy_(rwkv[b+'att.receptance.weight'])
    layer.tmix_k.weight.data.copy_(rwkv[b+'att.key.weight'])
    layer.tmix_v.weight.data.copy_(rwkv[b+'att.value.weight'])
    
    layer.ln_wkv.weight.data.copy_(rwkv[b+'ln1.weight'])
    layer.ln_wkv.bias.data.copy_(rwkv[b+'ln1.bias'])
    layer.ln_moe.weight.data.copy_(rwkv[b+'ln2.weight'])
    layer.ln_moe.bias.data.copy_(rwkv[b+'ln2.bias'])

model.embed.weight.data.copy_(rwkv['emb.weight'])
model.ln0.weight.data.copy_(rwkv['blocks.0.ln0.weight'])
model.ln0.bias.data.copy_(rwkv['blocks.0.ln0.bias'])
model.head.weight.data.copy_(rwkv['head.weight'])

d = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.embed.parameters()) - sum(p.numel() for p in model.head.parameters())
print(f'Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M total, {d/1e6:.1f}M mapped from RWKV')

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

start_step = 0
resume_path = os.path.join(CKPT_DIR, 'v6_latest.pt')
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step']
    print(f'Resumed from step {start_step}')

with open(CSV_PATH, 'a' if start_step > 0 else 'w', newline='') as f:
    if start_step == 0: f.write('step,loss,ppl,gn\n')

model.train()
pbar = tqdm(range(start_step, N_STEPS), initial=start_step, total=N_STEPS)
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    scheduler.step()

    if bi % 500 == 0: torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(loss=f'{loss.item():.2f}', ppl=f'{math.exp(min(loss.item(),20)):.1f}', gn=f'{gn:.1f}')
    if bi % 1000 == 0 or bi == N_STEPS - 1:
        ppl = math.exp(min(loss.item(), 20))
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{ppl:.4f},{gn:.1f}\n')
        torch.save({'step': bi+1, 'model': model.state_dict(), 'opt': opt.state_dict()},
                   resume_path + '.tmp')
        os.replace(resume_path + '.tmp', resume_path)

torch.save({'step': N_STEPS, 'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'v6_final.pt'))
print(f'\nDone - v6_final.pt')

print('\n--- Greedy (32 tokens) ---')
model.eval()
p = torch.randint(10, VOCAB, (1, 16), device=device)
with torch.no_grad():
    g = p.clone()
    for _ in range(16):
        l = model(g)
        g = torch.cat([g, l[:, -1].float().argmax(-1, keepdim=True)], dim=1)
    out = g[0].tolist()[16:]
    uni = len(set(out))
    print(f'  Unique: {uni}/16  {"DIVERSE" if uni > 4 else "COLLAPSED"}')
print('Done.')
