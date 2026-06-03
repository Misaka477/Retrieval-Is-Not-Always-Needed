"""Train confidence head: per-position gating for stateful denoiser."""
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
import torch, os, time, numpy as np
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_v7_demo import RWKV, args, RWKV_TOKENIZER

DEVICE = 'cuda'; D = 768
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
MODEL_PATH = os.path.join(BASE_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth')
os.makedirs(CKPT_DIR, exist_ok=True)

print('Loading backbone...')
sd = torch.load(MODEL_PATH, map_location='cpu')
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
model = RWKV(args).to(DEVICE); model.load_state_dict(sd,strict=False); model.eval()
for p in model.parameters(): p.requires_grad_(False)

# ── Stateful denoiser (frozen) + confidence head (trainable) ──
class StatefulDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = D
        self.input_proj = nn.Linear(D*2, D)
        self.log_A = nn.Parameter(torch.zeros(D))
        self.B = nn.Linear(D, D, bias=False)
        self.C = nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, D)
        self.out = nn.Linear(D, D)
        self.conf = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, h, cond, state=None):
        B = h.shape[0]
        if state is None:
            state = h.new_zeros(B, self.d_model)
        inp = self.input_proj(torch.cat([h, cond], -1))
        A = torch.sigmoid(self.log_A)
        state = A * state + self.B(inp)
        h_out = h + torch.sigmoid(self.gate(inp)) * self.out(self.C(state))
        return h_out, state

    def reset_state(self, B=1):
        return torch.zeros(B, self.d_model, device=DEVICE)

dn = StatefulDenoiser().to(DEVICE)
ckpt = torch.load(os.path.join(CKPT_DIR, 'dn_stateful_final.pt'), weights_only=False)['dn']
state = dn.state_dict()
for k,v in ckpt.items():
    if k in state and 'conf' not in k:
        state[k].copy_(v)
dn.load_state_dict(state, strict=False)
dn.train(); model.eval()

# Freeze denoiser, train only conf head
for n,p in dn.named_parameters():
    p.requires_grad_('conf' in n)
opt = torch.optim.AdamW(dn.conf.parameters(), lr=1e-3)

# ═══════════════════════════════════════════════════════════════
# Phase 0: Per-position confidence labels
#   label[t] = 1 if denoising UP TO t improves logprob of gt[t]
# ═══════════════════════════════════════════════════════════════
print('\nPhase 0: Per-position confidence labels (GT logprob delta)...')
trajs = torch.load(os.path.join(CKPT_DIR, 'ar_trajs.pt'), weights_only=False)
traj_h = trajs['h'].cuda()
traj_cond = trajs['cond'].cuda()
traj_gt = trajs['gt'].cuda()
N = traj_h.size(0); T = traj_h.size(1)

labels = torch.zeros(N, T, device=DEVICE)
with torch.no_grad():
    for i in tqdm(range(0, N, 256)):
        idx = slice(i, min(i+256, N))
        hb = traj_h[idx]; cb = traj_cond[idx]; gtb = traj_gt[idx]
        B = hb.size(0)

        # Compute per-position labels by running stateful denoiser forward once
        state = dn.reset_state(B)
        for t in range(T):
            h_out, state = dn(hb[:, t], cb[:, t], state)

            logits_before = model.head(hb[:, t].unsqueeze(0)).squeeze(0)
            logits_after = model.head(h_out.unsqueeze(0)).squeeze(0)

            lp_before = F.log_softmax(logits_before, -1)[range(B), gtb[:, t]]
            lp_after = F.log_softmax(logits_after, -1)[range(B), gtb[:, t]]

            labels[i.start:i.start+B, t] = (lp_after > lp_before).float()

p_improve = labels.mean().item()
print(f'  P(denoiser improves per-step logprob): {p_improve:.2%}')

# ── Phase 1: Train confidence head (per-position BCE) ──
print('\nPhase 1: Training confidence head...')
BSZ = 256; N_STEPS = 20000
for bi in tqdm(range(N_STEPS)):
    idx = torch.randint(0, N, (BSZ,), device=DEVICE)
    t = torch.randint(0, T, (BSZ,), device=DEVICE)
    hb = traj_h[idx, t]
    lb = labels[idx, t]

    conf = torch.sigmoid(dn.conf(hb)).squeeze(-1)
    loss = F.binary_cross_entropy(conf, lb)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(dn.conf.parameters(), 1.0); opt.step()

torch.save({'dn': dn.state_dict()}, os.path.join(CKPT_DIR, 'dn_conf_stateful.pt'))
print(f'Done. loss={loss.item():.4f}')

# ── Eval: AR vs AR+Dn vs AR+Conf ──
print('\n=== Comparison ===')
dn.eval()
tokenizer = RWKV_TOKENIZER(os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt'))

def bb(x):
    pad = (16-x.size(1)%16)%16
    if pad:
        xp = torch.cat([x,torch.zeros(1,pad,dtype=torch.long,device=DEVICE)],1)
        l,h = model(xp, return_h=True); return l[:,:x.size(1)], h[:,:x.size(1)]
    return model(x, return_h=True)

prompts = [
    "User: What is the capital of France?\n\nAssistant:",
    "The Eiffel tower is in the city of",
    "User: Who wrote Romeo and Juliet?\n\nAssistant:",
    "User: Write a short poem about a cat.\n\nAssistant:",
]

for prompt in prompts:
    p = torch.tensor([tokenizer.encode(prompt)]).to(DEVICE); plen = p.size(1)
    print(f'\nPrompt: {prompt}')
    for label, use_d, use_conf in [('AR',False,False),('AR+Dn',True,False),('AR+Conf',True,True)]:
        g = p.clone(); t0 = time.time()
        dn_state = dn.reset_state()
        with torch.no_grad():
            for _ in range(64):
                l,h = bb(g)
                if use_d:
                    cond = (torch.softmax(l*0.05,-1)@model.head.weight).reshape(-1,D)
                    h_raw = h.reshape(-1,D)
                    h_dn, dn_state = dn(h_raw, cond, dn_state)
                    if use_conf:
                        conf = torch.sigmoid(dn.conf(h_raw)).squeeze(-1)
                        h_mix = conf.unsqueeze(-1) * h_dn + (1-conf.unsqueeze(-1)) * h_raw
                        l = model.head(h_mix.unsqueeze(0))
                    else:
                        l = model.head(h_dn.unsqueeze(0))
                probs = torch.softmax(l[:,-1].float()/0.8,-1); probs[:,0]=0
                g = torch.cat([g,torch.multinomial(probs,1)],1)
        print(f'  {label} ({time.time()-t0:.0f}s): {repr(tokenizer.decode(g[0].tolist()[plen:]))}')
print('\nDone.')
