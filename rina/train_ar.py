"""Phase 0: collect AR trajectories → Phase 1: train stateful denoiser with per-step GT logprob."""
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
import torch, os, math, numpy as np, time
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_v7_demo import RWKV, args, RWKV_TOKENIZER

DEVICE = 'cuda'; D = 768; VOCAB = 65536
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
MODEL_PATH = os.path.join(BASE_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth')
DATA_PATH = os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy')
os.makedirs(CKPT_DIR, exist_ok=True)
tokenizer = RWKV_TOKENIZER(os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt'))

print(f'Loading backbone from {MODEL_PATH}...')
sd = torch.load(MODEL_PATH, map_location='cpu')
for k,v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32: sd[k] = v.float()
model = RWKV(args).to(DEVICE)
model.load_state_dict(sd, strict=False); model.eval()
for p in model.parameters(): p.requires_grad_(False)
print(f'Backbone: {sum(p.numel()/1e6 for p in model.parameters()):.1f}M')

# ═══════════════════════════════════════════════════════════════
# Phase 0: Collect AR trajectories (with GT tokens)
# ═══════════════════════════════════════════════════════════════
print('\n=== Phase 0: Collecting AR trajectories ===')
ids = torch.from_numpy(np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r'))
N_SEEDS = 20000
TRAJ_LEN = 32  # 16 seed tokens + 16 AR-generated

def run_model(x):
    pad = (16 - x.size(1) % 16) % 16
    if pad:
        xp = torch.cat([x, torch.zeros(1, pad, dtype=torch.long, device=DEVICE)], 1)
        l, h = model(xp, return_h=True)
        return l[:, :x.size(1)], h[:, :x.size(1)]
    return model(x, return_h=True)

trajectories = []
for _ in tqdm(range(N_SEEDS)):
    s = torch.randint(0, len(ids) - TRAJ_LEN - 1, (1,)).item()
    seed = ids[s:s+16].cuda().long().unsqueeze(0)

    traj_h, traj_cond, traj_gt = [], [], []
    g = seed.clone()
    with torch.no_grad():
        for pos in range(TRAJ_LEN - 16):
            l, h = run_model(g)
            h_t = h[0, -1].cpu()
            cond_t = (torch.softmax(l[0, -1]*0.05, -1) @ model.head.weight).cpu()
            gt_t = ids[s + 16 + pos].item()

            traj_h.append(h_t)
            traj_cond.append(cond_t)
            traj_gt.append(gt_t)

            probs = torch.softmax(l[0, -1].float() / 0.8, -1); probs[0] = 0
            nxt = torch.multinomial(probs, 1).unsqueeze(0)
            g = torch.cat([g, nxt], 1)

    trajectories.append({
        'h': torch.stack(traj_h),    # [16, D]
        'cond': torch.stack(traj_cond), # [16, D]
        'gt': torch.tensor(traj_gt),    # [16]
    })

traj_h = torch.stack([t['h'] for t in trajectories])    # [N, 16, D]
traj_cond = torch.stack([t['cond'] for t in trajectories])  # [N, 16, D]
traj_gt = torch.stack([t['gt'] for t in trajectories])      # [N, 16]

torch.save({'h': traj_h, 'cond': traj_cond, 'gt': traj_gt},
           os.path.join(CKPT_DIR, 'ar_trajs.pt'))
print(f'  Collected {len(trajectories)} trajectories x {TRAJ_LEN - 16} steps = {len(trajectories) * (TRAJ_LEN - 16)} states')
print(f'  h norm: {traj_h.norm(dim=-1).mean().item():.2f}')

# ═══════════════════════════════════════════════════════════════
# Phase 1: Train stateful denoiser with per-step GT logprob
# ═══════════════════════════════════════════════════════════════
print('\n=== Phase 1: Training stateful denoiser ===')

class StatefulDenoiser(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model * 2, d_model)

        self.log_A = nn.Parameter(torch.zeros(d_model))
        self.B = nn.Linear(d_model, d_model, bias=False)
        self.C = nn.Linear(d_model, d_model, bias=False)

        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, h, cond, state=None):
        B = h.shape[0]
        if state is None:
            state = h.new_zeros(B, self.d_model)

        inp = self.input_proj(torch.cat([h, cond], -1))
        A = torch.sigmoid(self.log_A)
        state = A * state + self.B(inp)
        correction = self.C(state)
        g = torch.sigmoid(self.gate(inp))
        h_out = h + g * self.out(correction)
        return h_out, state

    def reset_state(self, B=1, device='cuda'):
        return torch.zeros(B, self.d_model, device=device)

dn = StatefulDenoiser().to(DEVICE)
opt = torch.optim.AdamW(dn.parameters(), lr=3e-4)
TRAJ_LEN_GT = 16

data = torch.load(os.path.join(CKPT_DIR, 'ar_trajs.pt'), weights_only=False)
traj_h = data['h'].cuda()
traj_cond = data['cond'].cuda()
traj_gt = data['gt'].cuda()
N = traj_h.size(0)

N_EPOCHS = 10
BSZ = 64
BETA = 0.1
N_STEPS = N * TRAJ_LEN_GT // BSZ * N_EPOCHS

dn.train()
pbar = tqdm(range(N_STEPS))
step = 0
for epoch in range(N_EPOCHS):
    perm = torch.randperm(N)
    for start in range(0, N, BSZ):
        idx = perm[start:start+BSZ]
        B = idx.size(0)
        state = dn.reset_state(B)

        total_loss = 0
        for t in range(TRAJ_LEN_GT):
            h_t = traj_h[idx, t]
            cond_t = traj_cond[idx, t]
            gt_t = traj_gt[idx, t]

            h_out, state = dn(h_t, cond_t, state)

            logits = model.head(h_out.unsqueeze(0)).squeeze(0)
            ce_loss = F.cross_entropy(logits, gt_t)
            kl_loss = F.mse_loss(h_out, h_t)

            loss = ce_loss + BETA * kl_loss
            total_loss += loss

        total_loss /= TRAJ_LEN_GT
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(dn.parameters(), 10.0)
        opt.step()

        if step % 1000 == 0:
            with torch.no_grad():
                logprob_improve = 0
                for t in range(min(5, TRAJ_LEN_GT)):
                    h_before = traj_h[idx[:4], t]
                    gt_before = traj_gt[idx[:4], t]
                    lp_before = -F.cross_entropy(model.head(h_before.unsqueeze(0)).squeeze(0), gt_before)

                    s = dn.reset_state(4)
                    h_after = h_before.clone()
                    for tt in range(t+1):
                        h_after, s = dn(traj_h[idx[:4], tt], traj_cond[idx[:4], tt], s)
                    lp_after = -F.cross_entropy(model.head(h_after.unsqueeze(0)).squeeze(0), gt_before)
                    logprob_improve += (lp_after - lp_before).mean().item()

            pbar.set_postfix(ce=f'{ce_loss.item():.3f}', kl=f'{kl_loss.item():.3f}',
                             ce_avg=f'{total_loss.item():.3f}',
                             dL=f'{logprob_improve/5:.4f}')
            torch.save({'dn': dn.state_dict(), 'opt': opt.state_dict(), 'step': step},
                       os.path.join(CKPT_DIR, f'dn_stateful_{step}.pt'))

        step += 1
        pbar.update(1)

torch.save({'dn': dn.state_dict()}, os.path.join(CKPT_DIR, 'dn_stateful_final.pt'))
print(f'\nDone. final loss={total_loss.item():.4f}')

# ── Eval: AR vs AR+StatefulDenoiser ──
print('\n=== AR vs AR+StatefulDenoiser ===')
dn.eval()
prompts = [
    "User: What is the capital of France?\n\nAssistant:",
    "The Eiffel tower is in the city of",
    "User: Who wrote Romeo and Juliet?\n\nAssistant:",
    "User: Write a short poem about a cat.\n\nAssistant:",
]

for prompt in prompts:
    p = torch.tensor([tokenizer.encode(prompt)]).to(DEVICE); plen = p.size(1)
    print(f'\nPrompt: {prompt}')
    for label, use_d in [('AR', False), ('AR+Dn', True)]:
        g = p.clone(); t0 = time.time()
        dn_state = dn.reset_state()
        with torch.no_grad():
            for _ in range(64):
                l, h = run_model(g)
                if use_d:
                    cond = (torch.softmax(l*0.05,-1) @ model.head.weight).reshape(-1, D)
                    h_out, dn_state = dn(h.reshape(-1, D), cond, dn_state)
                    l = model.head(h_out.unsqueeze(0))
                probs = torch.softmax(l[:,-1].float()/0.8,-1); probs[:,0]=0
                g = torch.cat([g, torch.multinomial(probs,1)],1)
        print(f'  {label} ({time.time()-t0:.0f}s): {repr(tokenizer.decode(g[0].tolist()[plen:]))}')
print('\nDone.')
