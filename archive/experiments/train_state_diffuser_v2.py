"""State Diffuser: online training on official backbone states, eval at end."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm

device = 'cuda'; DM, VOCAB = 768, 65536
CKPT_PATH = 'rwkv7-g1d-0.1b-20260129-ctx8192.pth'
from rina.model_v7 import MoHEv7
model = MoHEv7(VOCAB, DM, n_layers=6, ckpt_path=CKPT_PATH).to(device)
model.eval()
for p in model.parameters(): p.requires_grad_(False)

class Diffuser(nn.Module):
    def __init__(self):
        super().__init__()
        D = DM
        self.norm = nn.LayerNorm(D * 2)
        self.net = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(),
            nn.Linear(D, D))
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, h, c):
        x = self.norm(torch.cat([h, c], dim=-1))
        return h + torch.tanh(self.gate) * self.net(x)  # scale down correction

diff = Diffuser().to(device)
opt = torch.optim.AdamW(diff.parameters(), lr=3e-5)
ids = torch.from_numpy(np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r'))
BSZ, SEQ, N_STEPS = 4, 128, 1000

diff.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    s = torch.randint(0, len(ids)-BSZ*SEQ, (1,)).item()
    x = ids[s:s+BSZ*SEQ].reshape(BSZ, SEQ).to(device)
    with torch.no_grad():
        logits, h = model(x, return_state=True)
    cond = (torch.softmax(logits*0.05, dim=-1) @ model.head.weight).view(-1, DM)
    hf = h.view(-1, DM)
    sigma = 0.01 + 0.09*torch.rand(1).item()  # small noise: diffuser learns tiny corrections
    h_noisy = hf + torch.randn_like(hf)*sigma
    h_pred = diff(h_noisy, cond)
    loss = F.mse_loss(h_pred, hf)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0); opt.step()
    if bi%200==0:
        pbar.set_postfix(loss=f'{loss.item():.4f}', sigma=f'{sigma:.2f}')
    torch.cuda.empty_cache()

torch.save({'diff': diff.state_dict()}, 'checkpoints/diff_official.pt')

# Eval: AR vs AR+Diff
model.eval(); diff.eval()
prompt = torch.randint(10, VOCAB, (1, 16), device=device)
for label, use_d in [('Pure AR', False), ('AR+Diff', True)]:
    g = prompt.clone()
    with torch.no_grad():
        for _ in range(8):
            logits, h = model(g, return_state=True)
            if use_d:
                h = diff(h.view(-1,DM), (torch.softmax(logits*0.05,-1)@model.head.weight).view(-1,DM)).view(1,-1,DM)
                logits = model.head(h)
            g = torch.cat([g, logits[:,-1].float().argmax(-1,keepdim=True)],1)
        out = g[0,16:].tolist()
        print(f'{label}: {len(set(out))}/8 unique')
print('Done.')
