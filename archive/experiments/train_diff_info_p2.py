import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model import WKV7Fn, _load_wkv7
_load_wkv7()

device = 'cuda'
DM, NP, N_EXP = 256, 384, 4
BSZ, SEQ = 16, 256
LR = 3e-4
P2_STEPS = 10000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'p2_info_conserve.csv')
os.makedirs(CKPT_DIR, exist_ok=True)

class GenExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(NP, DM) * 0.1)
    def forward(self, h):
        return F.relu(h @ self.patterns.T) @ self.patterns

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(65536, DM)
        self.embed_norm = nn.LayerNorm(DM)
        self.tmix_w = nn.Parameter(torch.randn(DM // 64, 64) * 0.1)
        self.tmix_r = nn.Linear(DM, DM, bias=False)
        self.tmix_k = nn.Linear(DM, DM, bias=False)
        self.tmix_v = nn.Linear(DM, DM, bias=False)
        self.tmix_a = nn.Linear(DM, DM, bias=False)
        self.experts = nn.ModuleList([GenExpert() for _ in range(N_EXP)])
        self.consolidate = nn.Linear(DM * N_EXP, DM)
        self.head = nn.Linear(DM, 65536, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x=None, emb=None, return_logits=False):
        if return_logits:
            return self.head(x)
        if emb is not None:
            B, T = emb.shape[:2]
        else:
            B, T = x.shape
            emb = self.embed_norm(self.embed(x))
        H, N = DM // 64, 64
        w = torch.exp(-torch.exp(self.tmix_w))
        r = self.tmix_r(emb).view(B,T,H,N).contiguous()
        k = self.tmix_k(emb).view(B,T,H,N).contiguous()
        v = self.tmix_v(emb).view(B,T,H,N).contiguous()
        a = self.tmix_a(emb).view(B,T,H,N).contiguous() * 0.01
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B,T,H,N).contiguous()
        h = WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B,T,DM)
        for _ in range(2):
            h_exps = torch.stack([e(h) for e in self.experts], dim=0)
            h_new = self.consolidate(h_exps.permute(1,2,0,3).reshape(B,T,DM*N_EXP))
            h = h + h_new
        return h

class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DM, 128), nn.ReLU(), nn.Linear(128, DM))
    def forward(self, h_cont):
        return self.net(h_cont.mean(1))

gen = Gen().to(device); rec = Reconstructor().to(device)

ckpt = torch.load('checkpoints/p1_diffusion.pt', map_location='cpu', weights_only=False)
gen.load_state_dict(ckpt['gen'])
print(f'Loaded p1_diffusion.pt into generator')

opt_g = torch.optim.AdamW(gen.parameters(), lr=LR)
opt_r = torch.optim.AdamW(rec.parameters(), lr=LR)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

with open(CSV_PATH, 'w', newline='') as f:
    f.write('step,loss_pos,loss_neg,gap,margin\n')

print(f'\n{"="*50}\nPhase 2: Info conservation (contrastive)\n{"="*50}\n')
gen.train(); rec.train()
pbar = tqdm(range(P2_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    ctx = x[:, :SEQ//4]
    cont = x[:, 3*SEQ//4:]

    h_ctx = gen(ctx)
    h_cont = gen(cont)
    target = h_ctx.mean(dim=1).detach()

    h_pos = rec(h_cont)
    loss_pos = F.mse_loss(h_pos, target)

    h_neg = rec(h_cont[torch.randperm(BSZ, device=device)])
    loss_neg = F.mse_loss(h_neg, target)

    margin = max(0.01, 1.0 - 0.99 * bi / P2_STEPS)
    loss = loss_pos + max(0, margin - loss_neg)

    opt_g.zero_grad(); opt_r.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(rec.parameters(), 1.0)
    opt_g.step(); opt_r.step()

    if bi % 500 == 0:
        torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(pos=f'{loss_pos.item():.4f}', neg=f'{loss_neg.item():.4f}',
                         gap=f'{loss_neg.item()-loss_pos.item():.4f}')
    if bi % 500 == 0 or bi == P2_STEPS - 1:
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss_pos.item():.6f},{loss_neg.item():.6f},{loss_neg.item()-loss_pos.item():.6f},{margin:.4f}\n')

torch.save({'gen': gen.state_dict(), 'rec': rec.state_dict()},
           os.path.join(CKPT_DIR, 'info_conserve_v3.pt'))
print(f'Done - info_conserve_v3.pt')
