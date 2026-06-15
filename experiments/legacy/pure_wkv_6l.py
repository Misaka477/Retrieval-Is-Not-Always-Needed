"""6层纯WKV + 标准AR CE — 训练管线验证。
不塌 = MoHE有问题。塌 = 训练管线有问题。"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 384; VOCAB = 65536; N_LAYERS = 12
SEQ, BSZ, LR, N_STEPS = 128, 4, 3e-4, 8000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
print(f'{N_LAYERS}L WKV, DM={DM}, V={VOCAB}, SEQ={SEQ}, BS={BSZ}')

class TimeMix(nn.Module):
    def __init__(self):
        super().__init__()
        C = DM
        self.tm_k = nn.Parameter(torch.ones(C))
        self.tm_v = nn.Parameter(torch.ones(C))
        self.tm_r = nn.Parameter(torch.ones(C))
        self.key = nn.Linear(C, C, bias=False)
        self.val = nn.Linear(C, C, bias=False)
        self.rec = nn.Linear(C, C, bias=False)
        self.out = nn.Linear(C, C, bias=False)
        for m in [self.key, self.val, self.rec, self.out]: nn.init.xavier_uniform_(m.weight, 0.5)

    def forward(self, x):
        B,T,C = x.shape
        mk,mv,mr = torch.sigmoid(self.tm_k), torch.sigmoid(self.tm_v), torch.sigmoid(self.tm_r)
        xk = x*mk + F.pad(x[:,1:],(0,0,0,1))*(1-mk)
        xv = x*mv + F.pad(x[:,1:],(0,0,0,1))*(1-mv)
        xr = x*mr + F.pad(x[:,1:],(0,0,0,1))*(1-mr)
        k,v,r = self.key(xk), self.val(xv), torch.sigmoid(self.rec(xr))
        out = []; h = k.new_zeros(B,C); w = k.new_zeros(B,C)
        for t_ in range(T):
            dec = torch.sigmoid(k[:,t_]*0.1+0.5)
            h = dec*h + k[:,t_]*v[:,t_]; w = dec*w + 1
            out.append(r[:,t_]*(h/(w+1e-8)))
        return self.out(torch.stack(out,dim=1))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        self.wkv = TimeMix()
        self.ffn = nn.Sequential(nn.Linear(DM,DM*4,bias=False),nn.GELU(),nn.Linear(DM*4,DM,bias=False))
    def forward(self,x):
        x = x + self.wkv(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

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

model = Model().to(DEVICE)
total = sum(p.numel() for p in model.parameters())
emb_w = model.emb.weight.numel()
head_w = model.head.weight.numel()
print(f'Params: {total/1e6:.2f}M (emb: {emb_w/1e6:.2f}M, head: {head_w/1e6:.2f}M, core: {(total-emb_w-head_w)/1e6:.2f}M)')

# Data
data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]
print(f'Data: {N/1e6:.0f}M tokens, range [{data.min()}-{data.max()}]')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N-seq-1, (bsz,))
    x = torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(DEVICE)
    return x

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train(); pbar = tqdm(range(N_STEPS)); t0 = time.time()

for step in pbar:
    x = get_batch(BSZ, SEQ)
    logits = model(x)
    ce = F.cross_entropy(logits[:,:-1].reshape(-1,VOCAB), x[:,1:].reshape(-1))
    opt.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    if step%1000==0:
        pbar.set_postfix(ce=f'{ce.item():.2f}', ppl=f'{torch.exp(ce).item():.0f}', lr=f'{sched.get_last_lr()[0]:.1e}')

elapsed = (time.time()-t0)/60
print(f'\nTrain: {elapsed:.1f}min, final CE: {ce.item():.2f} PPL: {torch.exp(ce).item():.0f}')

# Test generation
import sys; sys.path.insert(0, os.path.join(BASE_DIR, 'rina'))
from rwkv_v7_demo import RWKV_TOKENIZER
tk_path = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
tk = RWKV_TOKENIZER(tk_path) if os.path.exists(tk_path) else None

model.eval()
with torch.no_grad():
    for prompt in ["The capital of France is", "User: What is 2+2?\n\nAssistant:"]:
        if tk:
            ids = tk.encode(prompt)[:SEQ//2]
            x = torch.tensor([ids], device=DEVICE)
            for _ in range(40):  # generate 40 tokens
                logits = model(x)
                probs = torch.softmax(logits[:,-1].float()/0.8, -1)
                probs[0,0] = 0
                nxt = torch.multinomial(probs, 1)
                x = torch.cat([x, nxt], 1)
            print(f'Prompt: {prompt}')
            print(f'Gen:    {tk.decode(x[0].tolist())[:200]}')
            print()

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'pure_wkv_6l.pt'))
print('Saved.')
