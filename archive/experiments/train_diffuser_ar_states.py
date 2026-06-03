"""Phase 0: collect AR states → Phase 1: train denoiser on them (match inference distribution)."""
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
import torch, os, math, numpy as np, time
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_rwkv_official', 'RWKV-v7'))
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
# Phase 0: Collect AR states
# ═══════════════════════════════════════════════════════════════
print('\n=== Phase 0: Collecting AR states ===')
ids = torch.from_numpy(np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r'))
N_SEEDS = 5000

states, conds = [], []
for _ in tqdm(range(N_SEEDS)):
    s = torch.randint(0, len(ids) - 32, (1,)).item()
    g = ids[s:s+16].cuda().long().unsqueeze(0)
    with torch.no_grad():
        for pos in range(16):
            l, h = model(g, return_h=True)
            states.append(h[0, -1].cpu())
            c = (torch.softmax(l[0, -1]*0.05, -1) @ model.head.weight).cpu()
            conds.append(c)
            probs = torch.softmax(l[0, -1].float() / 0.8, -1); probs[0] = 0
            nxt = torch.multinomial(probs, 1).unsqueeze(0)
            g = torch.cat([g, nxt], 1)

states = torch.stack(states).cuda(); conds = torch.stack(conds).cuda()  # [80000, D]
torch.save({'states': states.cpu(), 'conds': conds.cpu()}, os.path.join(CKPT_DIR, 'ar_states.pt'))
print(f'  Collected {len(states)} states from AR trajectories')
print(f'  h norm: {states.norm(dim=-1).mean().item():.2f}')

# ═══════════════════════════════════════════════════════════════
# Phase 1: Train denoiser on AR states
# ═══════════════════════════════════════════════════════════════
print('\n=== Phase 1: Training denoiser on AR states ===')

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D*2, D*2), nn.GELU(),
            nn.Linear(D*2, D),
        )
    def forward(self, h, cond):
        return h + self.net(torch.cat([h, cond], -1))  # residual + direct

dn = Denoiser().to(DEVICE)
opt = torch.optim.AdamW(dn.parameters(), lr=1e-3)

# Load AR states
data = torch.load(os.path.join(CKPT_DIR, 'ar_states.pt'), weights_only=False)
h_pool = data['states'].cuda()
c_pool = data['conds'].cuda()

N_STEPS = 200000
BSZ = 128
dn.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    idx = torch.randint(0, h_pool.size(0), (BSZ,), device=DEVICE)
    h = h_pool[idx]
    c = c_pool[idx]
    sigma = 0.02 + 0.48 * torch.rand(1).item()
    hn = h + torch.randn_like(h) * sigma
    hp = dn(hn, c)
    loss = F.mse_loss(hp, h.detach(), reduction='sum') / BSZ
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(dn.parameters(), 100.0); opt.step()
    if bi % 5000 == 0:
        pbar.set_postfix(loss=f'{loss.item()/D:.4f}', wn=f'{sum(p.norm().item() for p in dn.parameters()):.1f}')
        torch.save({'dn': dn.state_dict()}, os.path.join(CKPT_DIR, f'dn_ar_{bi}.pt'))
    torch.cuda.empty_cache()

torch.save({'dn': dn.state_dict()}, os.path.join(CKPT_DIR, 'dn_ar_final.pt'))
print(f'\nDone. loss={loss.item()/D:.6f}')

# ── Eval: AR vs AR+Denoiser ──
print('\n=== AR vs AR+Denoiser ===')
dn.eval()
prompt = 'User: What is the capital of France?\n\nAssistant:'
p = torch.tensor([tokenizer.encode(prompt)]).to(DEVICE); plen = p.size(1)

def bb(x):
    pad = (16 - x.size(1)%16)%16
    if pad:
        xp = torch.cat([x, torch.zeros(1,pad,dtype=torch.long,device=DEVICE)],1)
        l,h = model(xp, return_h=True); return l[:,:x.size(1)], h[:,:x.size(1)]
    return model(x, return_h=True)

for label, use_d in [('AR', False), ('AR+Dn', True)]:
    g = p.clone(); t0 = time.time()
    with torch.no_grad():
        for _ in range(64):
            l, h = bb(g)
            if use_d:
                c = (torch.softmax(l*0.05,-1) @ model.head.weight).reshape(-1,D)
                h = dn(h.reshape(-1,D), c).reshape(1,-1,D)
                l = model.head(h)
            probs = torch.softmax(l[:,-1].float()/0.8,-1); probs[:,0]=0
            g = torch.cat([g, torch.multinomial(probs,1)],1)
    print(f'{label} ({time.time()-t0:.0f}s): {repr(tokenizer.decode(g[0].tolist()[plen:]))}')
print('Done.')
