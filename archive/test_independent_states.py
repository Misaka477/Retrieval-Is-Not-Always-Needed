"""
16M MoHE with independent h_i on FineWeb, 3000 steps + generation test.
Saves weights after training, runs generation on 5 prompts.
"""
import sys, os; sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from rina.mohe import MoHE
from rina.sample import sample

device = 'cuda'
VOCAB, DM, NP = 50257, 256, 512
BSZ, SEQ = 8, 128
LR = 3e-4
N_STEPS = 3000
LOG_PATH = 'checkpoints/mohe_16m_indep.csv'
WEIGHTS_PATH = 'checkpoints/mohe_16m_indep.pt'

tok = Tokenizer.from_pretrained('gpt2')

# Train or load
if os.path.exists(WEIGHTS_PATH):
    print('Loading saved weights...')
    model = MoHE(VOCAB, DM, NP, n_experts=4, aux_loss_weight=0.5, route_noise=0.2, topk=2).to(device)
    sd = torch.load(WEIGHTS_PATH, map_location=device)
    for k in list(sd.keys()):
        if k.startswith('prev_route') or '_batch_' in k:
            del sd[k]
    model.load_state_dict(sd, strict=False)
else:
    ids = torch.from_numpy(np.load('checkpoints/mohe_fw.npy', mmap_mode='r'))
    print(f'Training on {len(ids):,} tokens...')
    model = MoHE(VOCAB, DM, NP, n_experts=4, aux_loss_weight=0.5, route_noise=0.2, topk=2).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200))
    with open(LOG_PATH, 'w', newline='') as f:
        f.write('step,ppl,loss,lr,exp_sim,gate_ratio,grad_norm,aux_loss\n')
    model.train()
    total_loss = 0.0
    nb = (len(ids)-1)//(BSZ*SEQ)
    perm = torch.randperm(nb)
    pbar = tqdm(range(min(nb, N_STEPS)), desc='16M indep')
    for bi in pbar:
        start = perm[bi] * BSZ * SEQ
        x = ids[start:start+BSZ*SEQ].view(BSZ, SEQ).to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), x.view(-1), label_smoothing=0.1)
        loss = loss + getattr(model, '_last_aux_loss', 0.0)
        loss.backward()
        model.finish_training_step()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step(); scheduler.step()
        total_loss += loss.item()
        if bi % 200 == 199 or bi == min(nb, N_STEPS)-1:
            ppl = float(torch.exp(torch.tensor(total_loss/(bi+1))))
            exp_sim = getattr(model, '_last_exp_sim', 0.0)
            gr = getattr(model, '_gate_ratio', 0.0)
            aux = getattr(model, '_last_aux_loss', 0.0)
            lr_now = float(opt.param_groups[0]['lr'])
            with open(LOG_PATH, 'a', newline='') as f:
                f.write(f'{bi+1},{ppl:.1f},{loss.item():.2f},{lr_now:.2e},{exp_sim:.4f},{gr:.4f},{gn:.4f},{aux:.6f}\n')
            pbar.set_postfix(ppl=f'{ppl:.1f}', gn=f'{gn:.3f}', sim=f'{exp_sim:.3f}', gr=f'{gr:.2f}')
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f'Saved {WEIGHTS_PATH}')

# Generation test
model.eval()
prompts = [
    'The meaning of life is',
    'def hello_world():',
    'In the beginning, God created',
    'The key to intelligence is',
]
for p in prompts:
    ids = tok.encode(p).ids[:8]
    gen = ids[:]
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([gen[-64:]], device=device)
            logits = model(inp)[0, -1, :]
            gen.append(sample(logits, temp=0.7, top_p=0.9).item())
    text = tok.decode(gen)
    print(f'\n{p}\n{text[:300]}')
print('\nDONE')
