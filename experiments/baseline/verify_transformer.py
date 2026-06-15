"""标准 Transformer + GPT2词表 — 验证训练管线。"""
import os, sys, time, math, json
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

device = 'cuda'
VOCAB = 50257; DM = 384; N_LAYERS = 6; N_HEADS = 6
SEQ, BSZ, LR = 256, 8, 3e-4
N_STEPS = 10000
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CSV_PATH = os.path.join(CKPT_DIR, 'transformer_log.csv')

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.pos = nn.Parameter(torch.randn(1, SEQ, DM) * 0.02)
        self.ln_in = nn.LayerNorm(DM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DM, nhead=N_HEADS, dim_feedforward=DM*4,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, N_LAYERS)
        self.ln_out = nn.LayerNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, T = x.shape
        return self.head(self.ln_out(self.encoder(self.ln_in(self.emb(x) + self.pos[:, :T]))))

# Build text corpus from RWKV vocab file, tokenize with GPT2
print('Tokenizing training data with GPT2...')
tok = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer')
tok.pad_token = tok.eos_token

# Read RWKV vocab for text snippets
vocab_file = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
text_chunks = []
with open(vocab_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ')
            if len(parts) >= 1 and parts[0] and ord(parts[0][0]) >= 32:
                text_chunks.append(parts[0])

# Build training corpus from chunks
corpus = ' '.join(text_chunks[:50000])  # ~500K chars
corpus = corpus * 50  # repeat ~25M chars total
all_ids = tok.encode(corpus, max_length=10000000, truncation=True)
data = np.array(all_ids, dtype=np.int32)
N = data.shape[0]
print(f'Corpus: {len(corpus)} chars → {N} tokens')

def get_batch(bsz, seq):
    pos = np.random.randint(0, N-seq-1, (bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long() for p in pos]).to(device)

model = Transformer().to(device)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
model.train()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f: f.write('step,loss,val_ce,ppl,lr\n')

pbar = tqdm(range(N_STEPS)); t0 = time.time()
for step in pbar:
    x = get_batch(BSZ, SEQ)
    loss = F.cross_entropy(model(x)[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            xv = get_batch(2, SEQ)
            ce_v = F.cross_entropy(model(xv)[:, :-1].reshape(-1, VOCAB), xv[:, 1:].reshape(-1)).item()
        model.train()
        pbar.set_postfix(loss=f'{loss.item():.2f}', ce_v=f'{ce_v:.2f}', ppl=f'{math.exp(ce_v):.0f}')
        with open(CSV_PATH, 'a') as f:
            f.write(f'{step},{loss.item():.4f},{ce_v:.4f},{math.exp(ce_v):.0f},{sched.get_last_lr()[0]:.2e}\n')
        torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'transformer.pt'))

print(f'Done in {(time.time()-t0)/60:.1f}min')

# Generation test
model.eval()
with torch.no_grad():
    x = get_batch(1, 32)
    for _ in range(40):
        p = torch.softmax(model(x)[:, -1].float() / 0.8, -1); p[0, 0] = 0
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    bg = set(); [bg.add((x[0, i].item(), x[0, i+1].item())) for i in range(x.size(1)-1)]
    print(f'Bigrams: {len(bg)}/{x.size(1)-1}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'transformer_final.pt'))
