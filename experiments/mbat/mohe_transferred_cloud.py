"""Pure 12L WKV cloud training. Uses archive's kernel loading approach."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from torch.utils.cpp_extension import load

device = 'cuda'
VOCAB, DM, N_LAYERS = 65536, 384, 12
N = 64; H = DM // N
BSZ, SEQ = 2, 512
LR = 3e-4
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Load CUDA kernel ──
print('Loading CUDA kernel...')
if not os.path.exists('kernels/wkv7_fp32.cu'):
    # Copy from archive if available
    import shutil
    for f in ['wkv7_fp32.cu', 'wkv7_fp32.cpp']:
        src = os.path.join('archive', f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join('kernels', f))

load(name='wind_backstepping',
     sources=['kernels/wkv7_fp32.cu', 'kernels/wkv7_fp32.cpp'],
     extra_cuda_cflags=['-res-usage', f'-D_C_={N}', f'-D_CHUNK_LEN_=16', '-O3'],
     verbose=False)
print('Kernel loaded.')

# ── Model ──
class TimeMix(nn.Module):
    def __init__(self):
        super().__init__(); C = DM
        self.tm_k = nn.Parameter(torch.ones(C)); self.tm_v = nn.Parameter(torch.ones(C))
        self.tm_r = nn.Parameter(torch.ones(C))
        self.key = nn.Linear(C, C, bias=False); self.val = nn.Linear(C, C, bias=False)
        self.rec = nn.Linear(C, C, bias=False); self.out = nn.Linear(C, C, bias=False)
        self.w0 = nn.Parameter(torch.zeros(H, N))
        for m in [self.key, self.val, self.rec, self.out]: nn.init.xavier_uniform_(m.weight, 1.0)
    
    def forward(self, x):
        B, T, C = x.shape
        mk, mv, mr = torch.sigmoid(self.tm_k), torch.sigmoid(self.tm_v), torch.sigmoid(self.tm_r)
        xk = x * mk + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mk)
        xv = x * mv + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mv)
        xr = x * mr + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mr)
        k, v, r = self.key(xk), self.val(xv), torch.sigmoid(self.rec(xr))
        
        r = r.view(B, T, H, N).contiguous()
        k = k.view(B, T, H, N).contiguous()
        v = v.view(B, T, H, N).contiguous()
        w = F.softplus(self.w0).view(1, 1, H, N).expand(B, T, H, N).contiguous()
        z = torch.zeros(B, T, H, N, device=x.device)
        a = torch.zeros(B, T, H, N, device=x.device)
        
        pad = (16 - T % 16) % 16
        if pad:
            r = F.pad(r, (0, 0, 0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad))
            w = F.pad(w, (0, 0, 0, 0, 0, 0, 0, pad))
            z = F.pad(z, (0, 0, 0, 0, 0, 0, 0, pad))
            a = F.pad(a, (0, 0, 0, 0, 0, 0, 0, pad))
        
        TT = r.size(1)
        y = torch.empty(B, TT, H, N, device=x.device)
        s = torch.zeros(B, H, (TT + 15) // 16, N, N, device=x.device)
        sa = torch.zeros(B, H, (TT + 15) // 16, N, N, device=x.device)
        torch.ops.wind_backstepping.forward(w, r, k, v, z, a, y, s, sa)
        
        return self.out(y[:, :T].contiguous().view(B, T, C))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Linear(DM, DM * 4, bias=False)
        self.v = nn.Linear(DM * 4, DM, bias=False)
    def forward(self, x): return self.v(torch.relu(self.k(x)) ** 2)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        self.wkv = TimeMix(); self.ffn = FFN()
    def forward(self, x): x = x + self.wkv(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))

# ── Data ──
data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N_T = data.shape[0]; print(f'Data: {N_T/1e6:.0f}M tokens')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N_T - seq - 1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = Model().to(device)
t = sum(p.numel() for p in model.parameters())
e = model.emb.weight.numel(); h_ = model.head.weight.numel()
print(f'Params: {t/1e6:.2f}M (emb: {e/1e6:.2f}M head: {h_/1e6:.2f}M core: {(t-e-h_)/1e6:.2f}M)')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)
scaler = torch.cuda.amp.GradScaler()

model.train()
pbar = tqdm(range(10000))
t0 = time.time()

for step in pbar:
    x = get_batch(BSZ, SEQ)
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update(); sched.step()
    
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv = get_batch(1, SEQ)
            lv = model(xv)
            ppl = torch.exp(F.cross_entropy(lv[:, :-1].reshape(-1, VOCAB), xv[:, 1:].reshape(-1))).item()
        model.train()
        pbar.set_postfix(ce=f'{loss.item():.2f}', ppl=f'{ppl:.0f}', lr=f'{sched.get_last_lr()[0]:.2e}')
        torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'pure_12l_cloud.pt'))

# ── Generation test ──
print('\n=== Generation Test ===')
model.eval()
with torch.no_grad():
    x = get_batch(1, 32)
    for _ in range(40):
        logits = model(x)
        probs = torch.softmax(logits[:, -1].float() / 0.8, -1)
        probs[0, 0] = 0
        x = torch.cat([x, torch.multinomial(probs, 1)], 1)
    
    tok_path = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
    if os.path.exists(tok_path):
        with open(tok_path, encoding='utf-8') as f:
            tok = {i: l.split(' ')[0] for i, l in enumerate(f) if l.strip()}
        print(''.join(tok.get(int(i), '?') for i in x[0].tolist()))
    else:
        print(f'T[{x[0,0].item()}] {" ".join(str(i.item()) for i in x[0, :20])}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'pure_12l_final.pt'))
print('Done.')
