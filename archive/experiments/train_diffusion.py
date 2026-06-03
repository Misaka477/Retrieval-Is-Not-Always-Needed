"""Diffusion training from scratch — no CE, no AR. MSE on embeddings + routing."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_v5 import MoHERWKV_V5, AttractorExpertV5
from rina.model import WKV7Fn, _load_wkv7

_load_wkv7()
device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
BSZ, SEQ = 8, 512
LR = 2e-4; N_STEPS = 20000; SAVE_EVERY = 2000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CSV_PATH = os.path.join(CKPT_DIR, 'mohe_diffusion.csv')
RESUME_CKPT = os.path.join(CKPT_DIR, 'mohe_diffusion_latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print("Initializing from transferred checkpoint (diffusion fine-tune)...")
model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.0, topk=2).to(device)

# load transferred weights (align shapes: skip V5-specific layers)
init = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
s = init.get('model', init.get('model_state_dict', init))
msd = model.state_dict()
sd = {}
for k, v in s.items():
    if k in msd and v.shape == msd[k].shape:
        sd[k] = v
model.load_state_dict(sd, strict=False)
print(f'Loaded transferred weights: {len(sd)}/{len(msd)} keys matched')

opt = torch.optim.AdamW(model.parameters(), lr=LR)

start_step = 0
if os.path.exists(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step']
    print(f'Resumed step {start_step}')
else:
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('step,mse,gn,lr\n')

print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

model.train()
pbar = tqdm(range(start_step, N_STEPS), initial=start_step, total=N_STEPS)
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()

    emb = model.embed_norm(model.embed(x))
    B, T, D = emb.shape; H, N = D // 64, 64

    # add noise to EMBEDDING (not hidden state)
    sigma = 0.01 + 0.49 * 0.5 * (1 + math.cos(math.pi * (bi % N_STEPS) / N_STEPS))
    emb_noisy = emb + torch.randn_like(emb) * sigma

    # WKV - shared weights for clean and noisy paths
    w = torch.exp(-torch.exp(model.tmix_w))
    w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()

    def wkv_forward(e):
        r = model.tmix_r(e).view(B, T, H, N).contiguous()
        k = model.tmix_k(e).view(B, T, H, N).contiguous()
        v = model.tmix_v(e).view(B, T, H, N).contiguous()
        a = model.tmix_a(e).view(B, T, H, N).contiguous() * 0.01
        return WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B, T, D)

    def depth_forward(h_in):
        h = h_in; rw = None
        for depth in range(3):
            route_raw = (model.router(h) + model.router_bias) * 3.0
            rw = torch.softmax(route_raw, dim=-1)
            h_exps = torch.stack([e(h, emb)[0] for e in model.experts], dim=0)
            h_exps = model.expert_norm(h_exps.permute(1,2,0,3).reshape(-1, D)).reshape(B, T, model.n_experts, D)
            if model.topk > 0 and model.topk < model.n_experts:
                _, inds = rw.topk(model.topk, dim=-1)
                mask = torch.zeros(B, T, model.n_experts, device=device).scatter_(-1, inds, 1)
                h_exps = h_exps * mask.unsqueeze(-1)
            h = model.consolidate_norm(model.consolidate(h_exps.reshape(B, T, model.n_experts * D)))
        return h, rw

    # Teacher: clean path (target)
    with torch.no_grad():
        h_clean = wkv_forward(emb)
        h_ref, _ = depth_forward(h_clean)

    # Student: noisy path
    h_noisy_wkv = wkv_forward(emb_noisy)
    h_pred, route_weights = depth_forward(h_noisy_wkv)

    # MSE: noisy forward should produce same output as clean forward
    loss_mse = F.mse_loss(h_pred, h_ref.detach())

    # Routing regularization (prevent collapse)
    rw = route_weights
    route_ent = -(rw * torch.log(rw.clamp(min=1e-10))).sum(-1).mean()
    loss = loss_mse - 0.01 * route_ent

    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()

    if bi % 200 == 0:
        lr_val = LR * min(1.0, bi / 1000) if bi < 1000 else LR
        for g in opt.param_groups: g['lr'] = lr_val
        torch.cuda.empty_cache()

    pbar.set_postfix(mse=f'{loss_mse.item():.4f}', gn=f'{gn:.0f}')

    if bi % 2000 == 1999 or bi == N_STEPS - 1:
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss_mse.item():.6f},{gn:.1f},{opt.param_groups[0]["lr"]:.2e}\n')
        torch.save({'step': bi+1, 'model': model.state_dict(), 'opt': opt.state_dict()},
                   RESUME_CKPT + '.tmp')
        os.replace(RESUME_CKPT + '.tmp', RESUME_CKPT)
        torch.cuda.empty_cache()
        print(f'\n  Saved {bi+1}: mse={loss_mse.item():.6f}')

# final save
torch.save({'step': N_STEPS, 'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'mohe_diffusion_final.pt'))
print('Done → mohe_diffusion_final.pt')
