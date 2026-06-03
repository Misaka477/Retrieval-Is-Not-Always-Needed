"""State Diffuser: freeze official WKV7 backbone, train only the state denoiser."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm

# Force kernel compilation
from rina.model import WKV7Fn, _load_wkv7
_load_wkv7()

from rina.model_v7 import MoHEv7, WKV7_Official

device = 'cuda'
DM = 768
BSZ, SEQ = 8, 512
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
CKPT_PATH = 'rwkv7-g1d-0.1b-20260129-ctx8192.pth'
CSV_PATH = os.path.join(CKPT_DIR, 'state_diffuser.csv')

# ── Load frozen backbone (no MoE, just WKV7 layers × 6) ──
class WKV7Backbone(nn.Module):
    """6× WKV7 layers with pretrained weights. No MoE — just pure state generator."""
    def __init__(self, dm, ckpt_path):
        super().__init__()
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        embed_w = sd.pop('emb.weight')
        if embed_w.dtype != torch.float32: embed_w = embed_w.float()
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        for k in list(sd.keys()):
            if not isinstance(sd[k], torch.Tensor): continue
            if sd[k].dtype != torch.float32:
                sd[k] = sd[k].float()
        self.ln0 = nn.LayerNorm(dm)
        self.ln0.weight.data.copy_(sd['blocks.0.ln0.weight'])
        self.ln0.bias.data.copy_(sd['blocks.0.ln0.bias'])
        
        from rina.model_v7 import WKV7_Official
        self.wkv_layers = nn.ModuleList()
        for i in range(6):
            off_i = i * 2
            b = f'blocks.{off_i}.'
            layer = WKV7_Official(dm, i)
            # Load all WKV params from official checkpoint
            for k in ['x_r','x_w','x_k','x_v','x_a','x_g']:
                getattr(layer, k).data.copy_(sd[b+f'att.{k}'])
            for k in ['w0','w1','w2','a0','a1','a2','v0','v1','v2','g1','g2','k_k','k_a','r_k']:
                getattr(layer, k).data.copy_(sd[b+f'att.{k}'])
            layer.receptance.weight.data.copy_(sd[b+'att.receptance.weight'])
            layer.key.weight.data.copy_(sd[b+'att.key.weight'])
            layer.value.weight.data.copy_(sd[b+'att.value.weight'])
            layer.output.weight.data.copy_(sd[b+'att.output.weight'])
            layer.ln_x.weight.data.copy_(sd[b+'att.ln_x.weight'])
            layer.ln_x.bias.data.copy_(sd[b+'att.ln_x.bias'])
            for p in layer.parameters(): p.requires_grad_(False)
            self.wkv_layers.append(layer)
            
        self.ln_out = nn.LayerNorm(dm)
        self.ln_out.weight.data.copy_(sd['ln_out.weight'])
        self.ln_out.bias.data.copy_(sd['ln_out.bias'])
        self.head = nn.Linear(dm, 65536, bias=False)
        self.head.weight.data.copy_(sd['head.weight'])
        for p in [self.ln_out.parameters(), self.head.parameters()]:
            for pp in p: pp.requires_grad_(False)
        
    def forward(self, x):
        h = self.ln0(self.embed(x))
        v_first = torch.empty_like(h)
        for layer in self.wkv_layers:
            h, v_first = layer(h, v_first)
        h = self.ln_out(h)
        return self.head(h), h  # (logits, state)

# ── State Diffuser ──
class StateDiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        D = DM
        self.net = nn.Sequential(
            nn.Linear(D * 2, D * 2), nn.GELU(),
            nn.Linear(D * 2, D),
        )
    def forward(self, h_noisy, condition):
        return self.net(torch.cat([h_noisy, condition], dim=-1))

# ── Load backbone ──
print('Loading frozen WKV7 backbone...')
backbone = WKV7Backbone(DM, CKPT_PATH).to(device)
print(f'Backbone: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M (frozen)')
backbone.eval()

# ── Diffuser ──
diffuser = StateDiffuser().to(device)
opt = torch.optim.AdamW(diffuser.parameters(), lr=1e-4)
print(f'Diffuser: {sum(p.numel() for p in diffuser.parameters())/1e3:.1f}K')

# ── Data ──
ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

N_STEPS = 2000
with open(CSV_PATH, 'w', newline='') as f:
    f.write('step,loss,mse,sigma\n')

diffuser.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0] * BSZ * SEQ
    x = ids[s:s + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)

    with torch.no_grad():
        logits, h_clean = backbone(x)  # h: [B, SEQ, DM]

    # Corrupt state with noise
    sigma = 0.05 + 0.3 * torch.rand(1).item()
    noise = torch.randn_like(h_clean) * sigma
    h_noisy = h_clean + noise

    # Condition: use softmax output projected to DM
    probs = torch.softmax(logits * 0.1, dim=-1)  # temper softmax to prevent extreme
    cond = probs @ backbone.head.weight.detach()

    # Denoise
    h_pred = diffuser(h_noisy, cond)
    loss = F.mse_loss(h_pred, h_clean)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffuser.parameters(), 1.0)
    opt.step()

    if bi % 200 == 0:
        pbar.set_postfix(loss=f'{loss.item():.4f}', sigma=f'{sigma:.2f}')
    if bi % 500 == 0 or bi == N_STEPS - 1:
        with torch.no_grad():
            mse_eval = F.mse_loss(diffuser(h_noisy, cond), h_clean).item()
        with open(CSV_PATH, 'a', newline='') as f:
            f.write(f'{bi+1},{loss.item():.6f},{mse_eval:.6f},{sigma:.4f}\n')

torch.save({'diffuser': diffuser.state_dict()}, os.path.join(CKPT_DIR, 'state_diffuser.pt'))
print(f'\nDiffuser trained — state_diffuser.pt')

# ── Quick inference comparison: AR vs AR+Diffusion ──
print(f'\n{"="*50}\nInference comparison\n{"="*50}')
backbone.eval()
diffuser.eval()

prompt = torch.randint(10, 65536, (1, 16), device=device)

for label, use_diffuser in [('Pure AR', False), ('AR+StateDiff', True)]:
    print(f'\n--- {label} (24 tokens) ---')
    with torch.no_grad():
        g = prompt.clone()
        for _ in range(8):
            logits, h = backbone(g)
            if use_diffuser:
                cond = torch.softmax(logits, dim=-1) @ backbone.head.weight.T
                h = diffuser(h, cond)
                logits = backbone.head(h)
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            g = torch.cat([g, nxt], dim=1)
        out = g[0].tolist()[16:]
        uni = len(set(out))
        print(f'  Unique: {uni}/8  {"DIVERSE" if uni > 3 else "COLLAPSED"}')

print('\nDone.')
