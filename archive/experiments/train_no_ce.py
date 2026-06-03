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
P0_STEPS, P1_STEPS = 5000, 10000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_P0 = os.path.join(CKPT_DIR, 'p0_no_ce.csv')
CSV_P1 = os.path.join(CKPT_DIR, 'p1_no_ce.csv')
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
    def forward(self, x=None, emb=None):
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

def uni_loss(h):
    h_n = F.normalize(h, dim=-1)
    cos = torch.bmm(h_n, h_n.transpose(1, 2))
    m = 1 - torch.eye(h.size(1), device=h.device).unsqueeze(0)
    return (cos * m).square().mean()

gen = Gen().to(device)
opt = torch.optim.AdamW(gen.parameters(), lr=LR)
print(f'Generator: {sum(p.numel() for p in gen.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

# ===== Phase 0: Token-level representation learning =====
print(f'\n{"="*50}\nPhase 0: Token-level Representation\n{"="*50}\n')
print(f'  Loss: cosine(h_i, emb_i) + uniformity + norm\n')

with open(CSV_P0, 'w', newline='') as f:
    f.write('step,loss_recon,uniformity,norm,gn\n')

gen.train()
pbar = tqdm(range(P0_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    emb = gen.embed_norm(gen.embed(x))
    h = gen(emb=emb)

    cos = F.cosine_similarity(h.view(-1, DM), emb.view(-1, DM).detach(), dim=-1)
    loss_recon = (1 - cos).mean()

    loss_uni = uni_loss(h)

    h_norm = h.norm(dim=-1).mean()
    loss_norm = (h_norm - 1.0).abs()

    loss = loss_recon + 0.05 * loss_uni + 0.1 * loss_norm

    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0).item()
    opt.step()

    if bi % 500 == 0:
        torch.cuda.empty_cache()
    if bi % 200 == 0:
        pbar.set_postfix(rec=f'{loss_recon.item():.4f}', norm=f'{h_norm.item():.2f}', gn=f'{gn:.1f}')
    if bi % 500 == 0 or bi == P0_STEPS - 1:
        with open(CSV_P0, 'a', newline='') as f:
            f.write(f'{bi+1},{loss_recon.item():.6f},{loss_uni.item():.6f},{h_norm.item():.4f},{gn:.1f}\n')

torch.save({'gen': gen.state_dict()}, os.path.join(CKPT_DIR, 'p0_no_ce.pt'))
print(f'Phase 0 done - p0_no_ce.pt (recon={loss_recon.item():.4f})')

# ===== Phase 1: Cross-document contrastive =====
print(f'\n{"="*50}\nPhase 1: Cross-document Contrastive\n{"="*50}\n')
print(f'  Same document: maximize cos_sim(cont_j, ctx_k) across positions')
print(f'  Different docs: minimize (no direct constraint, uniformity handles it)')
print(f'  Gen directly trained to produce document-specific h\n')

with open(CSV_P1, 'w', newline='') as f:
    f.write('step,loss_pos,loss_neg,gap,uniformity\n')

gen.train()
opt2 = torch.optim.AdamW(gen.parameters(), lr=LR * 0.3)
pbar = tqdm(range(P1_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    ctx = x[:, :SEQ//4]
    cont = x[:, 3*SEQ//4:]

    h_ctx = gen(ctx)
    h_cont = gen(cont)

    hc_n = F.normalize(h_cont, dim=-1)
    hx_n = F.normalize(h_ctx, dim=-1)

    cos_pos = torch.bmm(hc_n, hx_n.transpose(1, 2)).max(dim=-1).values
    loss_pos = (1 - cos_pos).mean()

    perm_b = torch.randperm(BSZ, device=device)
    hc_shuf = hc_n[perm_b]
    cos_neg = torch.bmm(hc_shuf, hx_n.transpose(1, 2)).max(dim=-1).values
    loss_neg = (1 - cos_neg).mean()

    margin = max(0.01, 0.2 - 0.19 * bi / P1_STEPS)
    loss = loss_pos + max(0, margin + loss_pos.detach() - loss_neg)

    loss_uni = uni_loss(h_ctx) + uni_loss(h_cont)
    loss = loss + 0.02 * loss_uni

    loss_norm = (h_ctx.norm(dim=-1).mean() - 1.0).abs() + (h_cont.norm(dim=-1).mean() - 1.0).abs()
    loss = loss + 0.05 * loss_norm

    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    opt2.step()

    if bi % 500 == 0:
        torch.cuda.empty_cache()
    if bi % 200 == 0:
        gap = loss_neg.item() - loss_pos.item()
        pbar.set_postfix(pos=f'{loss_pos.item():.4f}', neg=f'{loss_neg.item():.4f}',
                         gap=f'{gap:.4f}')
    if bi % 500 == 0 or bi == P1_STEPS - 1:
        with open(CSV_P1, 'a', newline='') as f:
            f.write(f'{bi+1},{loss_pos.item():.6f},{loss_neg.item():.6f},{loss_neg.item()-loss_pos.item():.6f},{loss_uni.item():.6f}\n')

torch.save({'gen': gen.state_dict()}, os.path.join(CKPT_DIR, 'gen_contrastive.pt'))
gap = loss_neg.item() - loss_pos.item()
print(f'Phase 1 done - gen_contrastive.pt')
print(f'  pos={loss_pos.item():.4f}, neg={loss_neg.item():.4f}, gap={gap:.4f}')
if gap > 0.02:
    print(f'  *** SUCCESS: gap > 0.02 — Gen produces doc-specific h ***')
elif gap > 0:
    print(f'  *** WEAK: gap positive but small ({gap:.4f}) ***')
else:
    print(f'  *** FAILURE: gap ≤ 0 — no doc-level structure ***')
