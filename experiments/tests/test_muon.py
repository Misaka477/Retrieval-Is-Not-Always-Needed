"""Muon vs AdamW 快速对比：等显卡空闲后跑"""
import torch, time
from model import GPTConfig, GPT

device = 'cuda'
conf = GPTConfig(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, block_size=256, bias=False)
data = torch.randint(0, 50257, (4, 256), device=device)

def run(use_muon, steps=200):
    m = GPT(conf).to(device)
    losses = []; t0 = time.time()
    if use_muon:
        from muon import MuonWithAuxAdam
        h = [p for n,p in m.named_parameters() if p.dim()>=2 and 'embed' not in n and 'head' not in n and 'ln_' not in n]
        o = [p for n,p in m.named_parameters() if not(p.dim()>=2 and 'embed' not in n and 'head' not in n and 'ln_' not in n)]
        opt = MuonWithAuxAdam([
            dict(params=h, use_muon=True, lr=0.02, weight_decay=0.01),
            dict(params=o, use_muon=False, lr=3e-4, betas=(0.9,0.95), weight_decay=0.01),
        ])
        name = 'Muon+AdamW'
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9,0.95))
        name = 'AdamW'
    for s in range(steps):
        _, l = m(data, data)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % 50 == 0: losses.append((s, l.item()))
    return losses[-1][1], time.time()-t0

print('=== Muon vs AdamW ===')
print('AdamW 200 steps...')
a, ta = run(False)
print(f'  CE={a:.4f}  {ta:.0f}s')
print('Muon+AdamW 200 steps...')
m, tm = run(True)
print(f'  CE={m:.4f}  {tm:.0f}s')
print(f'Difference: Muon {m:.4f} vs AdamW {a:.4f}')
