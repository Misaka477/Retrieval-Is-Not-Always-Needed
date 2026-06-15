"""验证训练管线：12层官方RWKV + AR CE。内核、数据、CSV全含。"""
import os, sys, time, types
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = 'cuda'
VOCAB, DM, N_LAYERS = 65536, 768, 12
SEQ, BSZ, LR = 256, 4, 3e-4
N_STEPS = 30000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'verify_log.csv')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_v7_demo import RWKV_Tmix_x070, RWKV_CMix_x070

args = types.SimpleNamespace()
args.n_layer = N_LAYERS; args.n_embd = DM; args.vocab_size = VOCAB
args.head_size_a = 64; args.dim_att = DM; args.dim_ffn = DM * 4

class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
        self.wkv = RWKV_Tmix_x070(args, i)
        self.ffn = RWKV_CMix_x070(args, i)
    def forward(self, x, v_first):
        xx, v_first = self.wkv(self.ln1(x), v_first)
        x = x + xx; x = x + self.ffn(self.ln2(x))
        return x, v_first

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.ln_out = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        nn.init.normal_(self.head.weight, mean=0, std=0.02)
    def forward(self, x):
        h = self.emb(x); v_first = torch.zeros(x.size(0), DM, device=x.device)
        for b in self.blocks: h, v_first = b(h, v_first)
        return self.head(self.ln_out(h))

data = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = data.shape[0]; print(f'Data: {N/1e6:.0f}M')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = Model().to(device)
total = sum(p.numel() for p in model.parameters())
print(f'Params: {total/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f: f.write('step,ce_train,ce_val,ppl,lr,grad_norm\n')

pbar = tqdm(range(N_STEPS)); t0 = time.time()
for step in pbar:
    x = get_batch(BSZ, SEQ)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
    opt.step(); sched.step()
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            xv = get_batch(1, SEQ); lv = model(xv)
            ce_v = F.cross_entropy(lv[:, :-1].reshape(-1, VOCAB), xv[:, 1:].reshape(-1)).item()
            ppl_v = torch.exp(torch.tensor(ce_v)).item()
        model.train()
        pbar.set_postfix(ce=f'{loss.item():.2f}', ppl=f'{ppl_v:.0f}')
        with open(CSV_PATH, 'a') as f:
            f.write(f'{step},{loss.item():.4f},{ce_v:.4f},{ppl_v:.0f},{sched.get_last_lr()[0]:.2e},{gn:.2f}\n')
        torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'verify.pt'))

print(f'Train: {(time.time()-t0)/60:.1f}min')

# Generation test
model.eval()
with torch.no_grad():
    x = get_batch(1, 32)
    for _ in range(40):
        p = torch.softmax(model(x)[:, -1].float() / 0.8, -1); p[0, 0] = 0
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    tf = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
    if os.path.exists(tf):
        with open(tf) as f: tok = {i: l.split(' ')[0] for i, l in enumerate(f) if l.strip()}
        print(''.join(tok.get(int(i), '?') for i in x[0].tolist()))
    bg = set(); [bg.add((x[0,i].item(), x[0,i+1].item())) for i in range(x.size(1)-1)]
    print(f'Bigrams: {len(bg)}/{x.size(1)-1}')
    print(f'Log: {CSV_PATH}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'verify_final.pt'))
