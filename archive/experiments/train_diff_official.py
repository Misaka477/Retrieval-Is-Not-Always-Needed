"""Train state diffuser on official RWKV backbone (pure Python WKV7, correct arch)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, types, time
from tqdm import tqdm
from rina.official_model import RWKV as OfficialRWKV

device = 'cuda'
VOCAB, DM = 65536, 768
BSZ, SEQ = 4, 512
N_STEPS = 5000
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'

# ── Load official backbone ──
args = types.SimpleNamespace(n_embd=DM, n_layer=12, vocab_size=VOCAB, head_size_a=64)
print('Loading backbone...')
sd = torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth', map_location='cpu', weights_only=False)
for k, v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
        sd[k] = v.float()

backbone = OfficialRWKV(args).to(device)
backbone.load_state_dict(sd, strict=False)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)
print(f'Backbone: {sum(p.numel()/1e6 for p in backbone.parameters()):.1f}M')

# ── Diffuser ──
class Diffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DM * 2)
        self.net = nn.Sequential(
            nn.Linear(DM * 2, DM * 2), nn.GELU(),
            nn.Linear(DM * 2, DM),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    def forward(self, h, cond):
        return h + self.net(self.norm(torch.cat([h, cond], -1)))

diffuser = Diffuser().to(device)
opt = torch.optim.AdamW(diffuser.parameters(), lr=1e-4)
print(f'Diffuser: {sum(p.numel()/1e3 for p in diffuser.parameters()):.1f}K')

# ── Data ──
ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)

# ── Training ──
diffuser.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    perm = torch.randperm(nb)
    s = perm[0].item() * BSZ * SEQ
    x = ids[s : s + BSZ * SEQ].view(BSZ, SEQ).to(device)

    with torch.no_grad():
        logits, h = backbone(x, return_h=True)

    sigma = 0.02 + 0.08 * torch.rand(1).item()
    noise = torch.randn_like(h) * sigma
    h_noisy = h + noise
    cond = torch.softmax(logits * 0.05, -1) @ backbone.head.weight

    h_pred = diffuser(h_noisy.reshape(-1, DM), cond.reshape(-1, DM)).reshape(BSZ, SEQ, DM)
    loss = F.mse_loss(h_pred, h)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffuser.parameters(), 1.0)
    opt.step()

    if bi % 1000 == 0:
        pbar.set_postfix(mse=f'{loss.item():.6f}')
        torch.save({'diffuser': diffuser.state_dict()}, os.path.join(CKPT_DIR, 'diff_official.pt'))
    torch.cuda.empty_cache()

torch.save({'diffuser': diffuser.state_dict()}, os.path.join(CKPT_DIR, 'diff_official.pt'))
print(f'\nTraining done.')

# ── Quick eval ──
diffuser.eval()
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
prompt = 'The Eiffel tower is in the city of'
p = torch.tensor([tok.encode(prompt)]).cuda()
prompt_len = p.size(1)

# pad to 16-aligned for kernel
def bb_padded(x):
    pad = (16 - x.size(1) % 16) % 16
    if pad:
        xp = torch.cat([x, torch.zeros(1, pad, dtype=torch.long, device=x.device)], 1)
        l, h = backbone(xp, return_h=True)
        return l[:, :x.size(1)], h[:, :x.size(1)]
    return backbone(x, return_h=True)

print(f'\nPrompt: {prompt}')
for label, use_d in [('AR', False), ('AR+Diff', True)]:
    g = p.clone()
    with torch.no_grad():
        for _ in range(32):
            l, h = bb_padded(g)
            if use_d:
                c = (torch.softmax(l * 0.05, -1) @ backbone.head.weight).reshape(-1, DM)
                h = diffuser(h.reshape(-1, DM), c).reshape(1, -1, DM)
                l = backbone.head(h)
            g = torch.cat([g, torch.multinomial(torch.softmax(l[:, -1].float() / 0.8, -1), 1)], 1)
    txt = tok.decode(g[0].tolist()[prompt_len:])
    print(f'{label}: {repr(txt[:100])}')
print('Done.')
