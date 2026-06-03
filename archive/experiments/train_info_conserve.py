"""Info conservation v2: contrastive objective, long gap between ctx and cont.
Generator must produce continuations whose hidden state can identify the original context.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model import WKV7Fn, _load_wkv7
_load_wkv7()

device = 'cuda'
DM, NP, N_EXP = 256, 384, 4
BSZ, SEQ = 16, 512
LR = 3e-4; N_STEPS = 10000
MARGIN = 0.1
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

class GenExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(NP, DM) * 0.02)
    def forward(self, h):
        scores = F.relu(h @ self.patterns.T)
        return scores @ self.patterns

class TinyGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(65536, DM)
        self.embed_norm = nn.LayerNorm(DM)
        self.tmix_w = nn.Parameter(torch.randn(DM // 64, 64) * 0.01)
        self.tmix_r = nn.Linear(DM, DM, bias=False)
        self.tmix_k = nn.Linear(DM, DM, bias=False)
        self.tmix_v = nn.Linear(DM, DM, bias=False)
        self.tmix_a = nn.Linear(DM, DM, bias=False)
        self.experts = nn.ModuleList([GenExpert() for _ in range(N_EXP)])
        self.consolidate = nn.Linear(DM * N_EXP, DM)
    def forward(self, x):
        B, T = x.shape; D = DM; H, N = D // 64, 64
        emb = self.embed_norm(self.embed(x))
        w = torch.exp(-torch.exp(self.tmix_w))
        r = self.tmix_r(emb).view(B,T,H,N).contiguous()
        k = self.tmix_k(emb).view(B,T,H,N).contiguous()
        v = self.tmix_v(emb).view(B,T,H,N).contiguous()
        a = self.tmix_a(emb).view(B,T,H,N).contiguous() * 0.01
        w4d = w.unsqueeze(0).unsqueeze(0).expand(B,T,H,N).contiguous()
        h = WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B,T,D)
        for depth in range(2):
            h_exps = torch.stack([e(h) for e in self.experts], dim=0)
            h_new = self.consolidate(h_exps.permute(1,2,0,3).reshape(B,T,D*N_EXP))
            h = h_new
        return h

class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DM, 128), nn.ReLU(),
            nn.Linear(128, DM)
        )
    def forward(self, h_cont):
        return self.net(h_cont.mean(1))

gen = TinyGen().to(device); rec = Reconstructor().to(device)
opt_g = torch.optim.AdamW(gen.parameters(), lr=LR)
opt_r = torch.optim.AdamW(rec.parameters(), lr=LR)
print(f'Generator: {sum(p.numel() for p in gen.parameters())/1e6:.2f}M')
print(f'Reconstructor: {sum(p.numel() for p in rec.parameters())/1e3:.0f}K')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

gen.train(); rec.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb); s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    ctx = x[:, :SEQ//4]     # 128 tokens
    cont = x[:, 3*SEQ//4:]  # last 128 tokens (far gap from ctx)

    h_ctx = gen(ctx)        # [B, 128, DM]
    h_cont = gen(cont)      # [B, 128, DM]
    h_ctx_target = h_ctx.mean(dim=1).detach()  # [B, DM]

    # Positive pair
    h_pred_pos = rec(h_cont)  # predict ctx from correct continuation
    loss_pos = F.mse_loss(h_pred_pos, h_ctx_target)

    # Negative pairs: shuffled continuations
    perm_neg = torch.randperm(BSZ, device=device)
    h_cont_neg = h_cont[perm_neg]
    h_pred_neg = rec(h_cont_neg)
    loss_neg = F.mse_loss(h_pred_neg, h_ctx_target)

    # Contrastive: margin anneals from 1.0 → 0.01 over training
    margin = 1.0 - 0.99 * min(1.0, bi / N_STEPS)
    loss = loss_pos + max(0, margin - loss_neg)

    opt_g.zero_grad(); opt_r.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(rec.parameters(), 1.0)
    opt_g.step(); opt_r.step()
    if bi % 500 == 0: torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(pos=f'{loss_pos.item():.4f}', neg=f'{loss_neg.item():.4f}',
                         gap=f'{loss_neg.item()-loss_pos.item():.4f}')

torch.save({'gen': gen.state_dict(), 'rec': rec.state_dict()},
           os.path.join(CKPT_DIR, 'info_conserve_mini.pt'))
print(f'Done → info_conserve_mini.pt')
