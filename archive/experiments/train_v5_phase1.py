"""V5 Phase 1: Self-supervised expert pretraining — masked reconstruction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina.model_v5 import MoHERWKV_V5
from rina.model import WKV7Fn, _load_wkv7

_load_wkv7()
device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
BSZ, SEQ = 4, 512
N_STEPS = 5000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

print("V5 Phase 1 — masked reconstruction (95% mask)...")
model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, aux_loss_weight=0.1, topk=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

model.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    emb = model.embed_norm(model.embed(x))
    B, T, D = emb.shape; H, N = D // 64, 64

    w = torch.exp(-torch.exp(model.tmix_w))
    r = model.tmix_r(emb).view(B, T, H, N).contiguous()
    k = model.tmix_k(emb).view(B, T, H, N).contiguous()
    v = model.tmix_v(emb).view(B, T, H, N).contiguous()
    a = model.tmix_a(emb).view(B, T, H, N).contiguous() * 0.01
    w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()
    h_clean = WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B, T, D)

    total_recon = 0.0
    mask_rate = 0.95
    for exp in model.experts:
        mask = torch.rand(B, T, D, device=device) > mask_rate
        h_masked = h_clean * mask.float()

        scores = torch.relu(h_masked @ exp.patterns.T)
        field = scores @ exp.patterns
        field = exp.proj(field)
        field = exp.field_mix(field)
        field = exp.norm(field)
        gate = torch.sigmoid(exp.slow_gate(torch.cat([h_masked, emb], dim=-1)))
        h_pred = h_masked + gate * field
        recon_loss = ((h_pred - h_clean) ** 2)[~mask].mean() if mask.any() else ((h_pred - h_clean) ** 2).mean()
        total_recon += recon_loss

    loss = total_recon / len(model.experts)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.experts.parameters(), 1.0)
    opt.step()
    if bi % 200 == 0: torch.cuda.empty_cache()
    pbar.set_postfix(recon=f'{loss.item():.4f}')

torch.save({'model': model.state_dict(), 'opt': opt.state_dict()},
           os.path.join(CKPT_DIR, 'mohe_v5_phase1.pt'))
print('Done → mohe_v5_phase1.pt')
