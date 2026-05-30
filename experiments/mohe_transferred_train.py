"""MoHE-RWKV transferred. RWKV tokenizer data. WKV full. Auto-save & resume."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.architectures.mohe_rwkv import MoHERWKV

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 4, 512
LR = 3e-4; N_STEPS = 90000
CKPT_DIR = 'checkpoints'
RESUME_CKPT = os.path.join(CKPT_DIR, 'mohe_transferred_latest.pt')
INIT_CKPT = os.path.join(CKPT_DIR, 'mohe_transferred_init.pt')
CSV_PATH = os.path.join(CKPT_DIR, 'mohe_transferred_v2.csv')
DATA_PATH = os.path.join(CKPT_DIR, 'mohe_fw_rwkv.npy')
SAVE_EVERY = 1500
EVAL_EVERY = 500

os.makedirs(CKPT_DIR, exist_ok=True)

model = MoHERWKV(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, route_noise=0.0, topk=2).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
def lr_fn(s):
    if s < 200: return s / 200
    p = (s - 200) / (N_STEPS - 200)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

start_step = 0
total_loss = 0.0

if os.path.exists(RESUME_CKPT):
    print(f'Resuming from {RESUME_CKPT} ...')
    state = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
    sd = state['model']
    for k in list(sd.keys()):
        if k.startswith('prev_route') or '_batch_' in k:
            del sd[k]
    model.load_state_dict(sd, strict=False)
    opt.load_state_dict(state['opt'])
    scheduler.load_state_dict(state['scheduler'])
    start_step = state['step']
    total_loss = state['total_loss']
    print(f'  Resumed at step {start_step}')
else:
    print('No resume checkpoint found, initializing from transferred weights ...')
    ckpt = torch.load(INIT_CKPT, map_location='cpu', weights_only=False)
    for k in list(ckpt.keys()):
        if k.startswith('prev_route') or '_batch_' in k:
            del ckpt[k]
    model.load_state_dict(ckpt, strict=False)
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,ppl,loss,lr,grad_norm,aux_loss,route_ent,exp_sim,val_ppl\n')

# break routing symmetry: amplify router init (overwritten by checkpoint)
model.router.weight.data.mul_(2.0)
model.router_bias.data = torch.randn_like(model.router_bias) * 0.5

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,ppl,loss,lr,grad_norm,aux_loss,route_ent,exp_sim,val_ppl\n')

print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
n_val = max(1, len(ids) // 100)
ids_train = ids[:-n_val]
ids_val = ids[-n_val:]
print(f'Train: {len(ids_train):,}, Val: {len(ids_val):,}')

@torch.no_grad()
def eval_val():
    model.eval()
    val_loss = 0.0; val_cnt = 0; steps = 0
    for s in range(0, len(ids_val) - BSZ * SEQ, BSZ * SEQ):
        x = ids_val[s:s+BSZ*SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
        val_loss += loss.item() * x.size(0)
        val_cnt += x.size(0)
        steps += 1
        if steps >= 100: break
    model.train()
    return float(torch.exp(torch.tensor(val_loss / max(val_cnt, 1))))

model.train()
nb = (len(ids_train)-1)//(BSZ*SEQ)
perm = torch.randperm(nb)
pbar = tqdm(range(start_step, min(nb, N_STEPS)), desc='MoHE-RWKV WKV',
            initial=start_step, total=min(nb, N_STEPS))
for bi in pbar:
    s = perm[bi] * BSZ * SEQ
    x = ids[s:s+BSZ*SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()
    logits = model(x)
    current_aux = getattr(model, '_last_aux_loss', 0.0)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss = loss + current_aux
    if torch.isnan(loss) or torch.isinf(loss):
        pbar.write(f'  NaN loss at step {bi+1}, skipping')
        scheduler.step()
        continue
    loss.backward()
    model.finish_training_step()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    if torch.isnan(torch.tensor(gn)) or any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
        pbar.write(f'  NaN grad at step {bi+1}, skipping optimizer')
        opt.zero_grad()
        scheduler.step()
        continue
    opt.step(); scheduler.step()
    total_loss += loss.item()
    if bi % EVAL_EVERY == EVAL_EVERY - 1:
        val_ppl = eval_val()
    if bi % SAVE_EVERY == SAVE_EVERY - 1 or bi == min(nb, N_STEPS) - 1:
        ppl = float(torch.exp(torch.tensor(total_loss/(bi + 1))))
        aux = current_aux
        route_ent = getattr(model, '_last_route_entropy', 0.0)
        if bi % EVAL_EVERY == EVAL_EVERY - 1:
            val_ppl = eval_val()
        with torch.no_grad():
            pts = [e.patterns.data for e in model.experts]
            sims = [(pts[i] @ pts[j].T).mean().item() for i in range(model.n_experts) for j in range(i+1, model.n_experts)]
            exp_sim = max(0.0, sum(sims) / len(sims)) if sims else 0.0
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{ppl:.1f},{loss.item():.2f},{opt.param_groups[0]["lr"]:.2e},{gn:.4f},{aux:.6f},{route_ent:.4f},{exp_sim:.4f},{val_ppl:.1f}\n')
        pbar.set_postfix(ppl=f'{ppl:.1f}', gn=f'{gn:.3f}')
        torch.save({
            'step': bi + 1,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'total_loss': total_loss,
        }, RESUME_CKPT + '.tmp')
        os.replace(RESUME_CKPT + '.tmp', RESUME_CKPT)
        torch.cuda.empty_cache()
        print(f'\n  Saved checkpoint at step {bi+1}')

print('DONE')
