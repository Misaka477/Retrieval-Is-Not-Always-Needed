"""SSD: Semantic State Diffusion on 12L WKV backbone.

Diffuse the full state sequence [B, T, DM] as continuous vectors.
No CE anywhere. Backbone is frozen, only diffuser + decoder trained.

Forward: h[0:T] → add noise → noisy_h[t]
Reverse: noisy_h[t] + timestep → diffuser → clean_h[t]
Loss: MSE(pred_clean, actual_clean)  ← 联合分布学习
Decode: denoised_state → embed_pred → nearest token via cos similarity
"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ['CUDA_HOME'] = '/home/aquama/miniconda3/envs/natalia'
os.environ['CPATH'] = '/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
os.environ['LD_LIBRARY_PATH'] = '/home/aquama/miniconda3/envs/natalia/lib'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 768; VOCAB = 65536

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
sys.path.insert(0, os.path.join(BASE_DIR, 'rina'))
from rwkv_v7_demo import RWKV, args

sd = torch.load(os.path.join(BASE_DIR, 'rwkv7-g1d-0.1b-20260129-ctx8192.pth'), map_location='cpu')
for k,v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32: sd[k] = v.float()
bk = RWKV(args).to(DEVICE)
bk.load_state_dict(sd, strict=False); bk.eval()
for p in bk.parameters(): p.requires_grad_(False)
print(f'Backbone: {sum(p.numel() for p in bk.parameters())/1e6:.1f}M')
E = bk.emb  # shorthand

# ═══════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════

data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]

def get_batch(bsz, seq_len):
    pos = np.random.randint(0, N - bsz * seq_len - 1)
    return torch.from_numpy(data[pos:pos+bsz*seq_len].copy()).long().view(bsz, seq_len).to(DEVICE)

@torch.no_grad()
def encode(x):
    """Get clean states [B, T, DM] from backbone."""
    pad = (16 - x.size(1) % 16) % 16
    if pad:
        xp = torch.cat([x, torch.zeros(x.size(0), pad, dtype=torch.long, device=DEVICE)], 1)
        _, h = bk(xp, return_h=True)
        return h[:, :x.size(1)]
    _, h = bk(x, return_h=True)
    return h

# ═══════════════════════════════════════════════════
# Diffusion components
# ═══════════════════════════════════════════════════

class NoiseSchedule:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    def __init__(self, T=1000):
        self.T = T
        s = 0.008
        t = torch.linspace(0, T, T+1)
        f = torch.cos((t/T + s) / (1 + s) * math.pi/2) ** 2
        self.alpha_bar = f / f[0]  # [T+1], ᾱ_t
        self.beta = torch.clip(1 - self.alpha_bar[1:] / self.alpha_bar[:-1], max=0.999)

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.beta = self.beta.to(device)
        return self

    def diffuse(self, h_0, t):
        """Add noise: h_t = sqrt(ᾱ_t) * h_0 + sqrt(1-ᾱ_t) * ε"""
        abar = self.alpha_bar[t.to(torch.long)].view(-1, 1, 1)
        eps = torch.randn_like(h_0)
        h_t = torch.sqrt(abar) * h_0 + torch.sqrt(1 - abar) * eps
        return h_t, eps

ns = NoiseSchedule(T=200).to(DEVICE)  # 200 steps for fast experiment

# ═══════════════════════════════════════════════════
# Diffuser: MLP with timestep conditioning
# ═══════════════════════════════════════════════════

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)

class Diffuser(nn.Module):
    def __init__(self, d_model=DM):
        super().__init__()
        t_dim = 128
        self.t_embed = SinusoidalEmbed(t_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model + t_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.5)

    def forward(self, h_t, t):
        """h_t: [B, T, DM]  t: [B]  →  pred_eps or pred_h0"""
        t_e = self.t_embed(t).unsqueeze(1).expand(-1, h_t.size(1), -1)
        return self.net(torch.cat([h_t, t_e], -1))

diff = Diffuser().to(DEVICE)
head = nn.Linear(DM, DM).to(DEVICE)  # state → embed
nn.init.xavier_uniform_(head.weight, 0.5)
print(f'Diffuser: {sum(p.numel() for p in diff.parameters())/1e3:.1f}K')
print(f'Head: {sum(p.numel() for p in head.parameters())/1e3:.1f}K')
params = list(diff.parameters()) + list(head.parameters())

# ═══════════════════════════════════════════════════
# Train: predict clean state from noisy state
# ═══════════════════════════════════════════════════

BSZ, SEQ, LR, STEPS = 8, 64, 3e-4, 5000
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
diff.train(); head.train()

print('\n=== Training SSD ===')
pbar = tqdm(range(STEPS))
for step in pbar:
    x = get_batch(BSZ, SEQ)
    with torch.no_grad():
        h0 = encode(x).detach()  # [B, T, DM] clean states

    t = torch.randint(1, ns.T, (BSZ,), device=DEVICE)
    h_t, eps = ns.diffuse(h0, t)

    eps_pred = diff(h_t, t)
    loss = F.mse_loss(eps_pred, eps)

    # + reconstruction loss: head(denoised_h) → match emb
    with torch.no_grad():
        abar = ns.alpha_bar[t.to(torch.long)].view(-1, 1, 1)
        h_pred = (h_t - torch.sqrt(1 - abar) * eps_pred) / torch.sqrt(abar)
    emb_pred = head(h_pred)
    emb_target = E(x)[:, :-1]  # use all but last as target
    # h_pred is [B, T, DM] but we need to align with positions
    h_pred_aligned = h_pred[:, :-1]
    emb_pred_aligned = head(h_pred_aligned)
    recon_loss = F.mse_loss(emb_pred_aligned, emb_target.detach())

    total_loss = loss + 0.1 * recon_loss

    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    sched.step()

    if step % 1000 == 0:
        with torch.no_grad():
            tx = get_batch(4, SEQ)
            th0 = encode(tx).detach()
            tt = torch.randint(1, ns.T, (4,), device=DEVICE)
            th_t, teps = ns.diffuse(th0, tt)
            tep = diff(th_t, tt)
            tmse = F.mse_loss(tep, teps).item()
            tabar = ns.alpha_bar[tt.to(torch.long)].view(-1, 1, 1)
            thp = (th_t - torch.sqrt(1-tabar)*tep) / torch.sqrt(tabar)
            tcos = F.cosine_similarity(thp.view(-1, DM), th0.view(-1, DM)).mean().item()
        pbar.set_postfix(loss=f'{loss.item():.4f}', recon=f'{recon_loss.item():.4f}',
                         tmse=f'{tmse:.4f}', tcos=f'{tcos:.3f}', lr=f'{sched.get_last_lr()[0]:.1e}')

print(f'\nDone.')  # Phase 1 done marker

# ═══════════════════════════════════════════════════
# Phase 2: Train translator (state → token, CE, separate)
# ═══════════════════════════════════════════════════

print('\n=== Phase 2: Translater ===')
diff.eval()
translator = nn.Sequential(
    nn.Linear(DM, DM), nn.GELU(),
    nn.Linear(DM, DM), nn.GELU(),
    nn.Linear(DM, VOCAB, bias=False),
).to(DEVICE)
for m in translator:
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.1)

opt_t = torch.optim.AdamW(translator.parameters(), lr=3e-4)
N_TRAIN = 50000
pbar = tqdm(range(N_TRAIN))
for step in pbar:
    x = get_batch(8, 64)
    with torch.no_grad():
        h = encode(x).detach()
    logits = translator(h[:, :-1])
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt_t.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(translator.parameters(), 5.0)
    opt_t.step()
    if step % 1000 == 0:
        pbar.set_postfix(ce=f'{ce.item():.2f}', ppl=f'{torch.exp(ce).item():.0f}')

# Eval on clean states
with torch.no_grad():
    tx = get_batch(8, 64)
    th0 = encode(tx).detach()
    tl = translator(th0[:, :-1])
    tce = F.cross_entropy(tl.reshape(-1, VOCAB), tx[:, 1:].reshape(-1))
    tppl = torch.exp(tce).item()
    print(f'\n  Clean states: CE={tce.item():.2f} PPL={tppl:.0f}')

    # Eval on denoised states
    h_noise = torch.randn_like(th0)
    h_cur = h_noise
    for t_ in range(ns.T, 0, -1):
        t = torch.full((h_cur.size(0),), t_, device=DEVICE, dtype=torch.long)
        ep = diff(h_cur, t)
        abar = ns.alpha_bar[t_]
        abar_p = ns.alpha_bar[t_-1] if t_ > 0 else torch.tensor(1.0, device=DEVICE)
        bt = 1 - abar / abar_p
        h_cur = (h_cur - bt / torch.sqrt(1 - abar) * ep) / torch.sqrt(1 - bt)
        if t_ > 1:
            h_cur = h_cur + torch.sqrt(bt) * torch.randn_like(h_cur)

    tl_d = translator(h_cur[:, :-1])
    dce = F.cross_entropy(tl_d.reshape(-1, VOCAB), tx[:, 1:].reshape(-1))
    dppl = torch.exp(dce).item()
    dacc = (tl_d.argmax(-1) == tx[:, 1:]).float().mean().item()
    print(f'  Denoised states: CE={dce.item():.2f} PPL={dppl:.0f} Acc={dacc*100:.1f}%')

torch.save({'diff': diff.state_dict(), 'head': head.state_dict(), 'translator': translator.state_dict()},
           os.path.join(CKPT_DIR, 'ssd_final.pt'))
print('\nSaved.')
print()
print('To generate: load ssd_final.pt, run DDPM loop, translator outputs logits.')
