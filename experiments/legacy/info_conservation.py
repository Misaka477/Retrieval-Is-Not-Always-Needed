"""Info Conservation v2: WKV + conservation loss on real text.

Key fixes from v1:
- Vocab 4096 (manageable, matches CANN phase scale)
- Strong conservation weight (0.1, up from 0.01)
- Input reconstruction: reconstruct entire sequence, not just first half
- Data: character-level text (no tokenizer dependency)

Hypothesis: If conservation loss improves state structure,
we should see state self-cos > random-cos ratio grow over 1.0.
"""
import os, sys, time, glob, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'
DM = 256
VOCAB = 4096
SEQ_LEN = 64
BSZ = 64
LR = 3e-4
N_STEPS = 10000
CONSV_WEIGHT = 0.1  # 10x stronger than v1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'DM={DM} V={VOCAB} SEQ={SEQ_LEN} BS={BSZ} WS={CONSV_WEIGHT}')

# ═══════════════════════════════════════════════════
# Data: character-level text from Gutenberg / samples
# ═══════════════════════════════════════════════════

def build_vocab_and_data():
    """Build char-level vocab from sample text or generate synthetic."""
    texts = []
    DATA_PATH = os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy')
    ids = np.load(DATA_PATH, mmap_mode='r')
    return np.random.default_rng(42).integers(0, 65535, size=100_000_000, dtype=np.int32)
    print(f'  Generated {len(data)} tokens')

# ═══════════════════════════════════════════════════
# WKV TimeMix (simplified, per-dim decay)
# ═══════════════════════════════════════════════════

class WKVLayer(nn.Module):
    def __init__(self):
        super().__init__()
        C = DM
        self.time_mix_k = nn.Parameter(torch.ones(C))
        self.time_mix_v = nn.Parameter(torch.ones(C))
        self.time_mix_r = nn.Parameter(torch.ones(C))

        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        for m in [self.key, self.value, self.receptance, self.output]:
            nn.init.xavier_uniform_(m.weight, 0.5)

    def forward(self, x, return_all_states=False):
        B, T, C = x.shape
        mk = torch.sigmoid(self.time_mix_k)
        mv = torch.sigmoid(self.time_mix_v)
        mr = torch.sigmoid(self.time_mix_r)

        xk = x * mk + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mk)
        xv = x * mv + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mv)
        xr = x * mr + F.pad(x[:, 1:], (0, 0, 0, 1)) * (1 - mr)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        states = [] if return_all_states else None
        h = k.new_zeros(B, C)
        w = k.new_zeros(B, C)
        out = []
        for t in range(T):
            decay = torch.sigmoid(k[:, t] * 0.1 + 0.5)
            h = decay * h + k[:, t] * v[:, t]
            w = decay * w + 1
            o = r[:, t] * (h / (w + 1e-8))
            out.append(o)
            if return_all_states:
                states.append(h.clone())

        if return_all_states:
            return torch.stack(out, dim=1), torch.stack(states, dim=1)
        return torch.stack(out, dim=1)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        C = DM
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        nn.init.xavier_uniform_(self.key.weight, 0.5)
        nn.init.xavier_uniform_(self.value.weight, 0.5)

    def forward(self, x):
        return self.value(torch.relu(self.key(x)) ** 2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DM)
        self.ln2 = nn.LayerNorm(DM)
        self.wkv = WKVLayer()
        self.ffn = FFN()

    def forward(self, x, return_all_states=False):
        if return_all_states:
            o, s = self.wkv(self.ln1(x), return_all_states=True)
            x = x + o
            x = x + self.ffn(self.ln2(x))
            return x, s
        x = x + self.wkv(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block() for _ in range(2)])  # 2 layers
        self.ln_out = nn.LayerNorm(DM)
        self.ce_head = nn.Linear(DM, VOCAB, bias=False)

        # Conservation: decode all positions from final state
        self.consv_start = nn.Linear(DM, DM)
        self.consv_step = nn.Linear(DM, DM)
        self.consv_norm = nn.LayerNorm(DM)
        self.consv_predict = nn.Linear(DM, DM)  # predict embed from state
        nn.init.normal_(self.consv_start.weight, std=0.02)
        nn.init.normal_(self.consv_step.weight, std=0.02)
        nn.init.normal_(self.consv_predict.weight, std=0.02)

    def forward(self, x, return_states=False):
        B, T = x.shape
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, return_all_states=return_states)
        h = self.ln_out(h)
        return self.ce_head(h), h

    def conservation_loss(self, x):
        """Reconstruct all positions' embeddings from the final state.
        Forces the final state to carry full sequence information."""
        B, T = x.shape
        emb = self.embed(x)

        # Get all states (not just final) from last block
        _, h = self.forward(x)  # [B, T, DM]; h is last-layer states
        h_final = h[:, -1:]  # [B, 1, DM]

        # Reconstruct: expand final state into full sequence
        h0 = self.consv_norm(h_final)
        h0 = torch.tanh(self.consv_start(h0))  # [B, 1, DM]

        recon = [h0]
        for _ in range(T - 1):
            h_next = recon[-1] + 0.1 * torch.tanh(self.consv_step(recon[-1]))
            recon.append(h_next)

        recon = torch.cat(recon, dim=1)  # [B, T, DM]
        recon = self.consv_predict(recon)

        return F.mse_loss(recon, emb.detach())

    def state_structure(self, x):
        """Measure state structure. High self-cos/random-cos ratio = structured."""
        B, T = x.shape
        with torch.no_grad():
            _, h = self.forward(x)
            hf = h.view(-1, DM)
            nrm = F.normalize(hf, dim=-1)
            sim_self = (nrm @ nrm.T).mean().item()
            idx = torch.randperm(B * T, device=x.device)
            sim_shuf = (nrm @ nrm[idx].T).mean().item()
            return sim_self, sim_shuf


# ═══════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════

token_ids = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = token_ids.shape[0]

# Map 65K tokens → 4K via frequency binning
from collections import Counter
sample_ids = token_ids[np.random.default_rng(42).integers(0, N, size=50000)]
freq = Counter(sample_ids.tolist())
top_vocab = [t for t, _ in freq.most_common(VOCAB)]
token_to_id = {t: i for i, t in enumerate(top_vocab)}
oov = VOCAB - 1  # last index
print(f'Vocab: {len(top_vocab)}/{VOCAB} tokens, OOV token={oov}, coverage: {sum(freq[t] for t in top_vocab)/sum(freq.values())*100:.1f}%')

model = Model().to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f'Params: {total/1e6:.2f}M')
print(f'  conservation: {sum(p.numel() for n,p in model.named_parameters() if "consv" in n)/1e3:.1f}K')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

model.train()
pbar = tqdm(range(N_STEPS))
step = 0
t0 = time.time()
tok_buf = np.zeros(BSZ * SEQ_LEN, dtype=np.int64)

for step in pbar:
    # Sample tokens and map to our vocab
    pos = np.random.randint(0, N - BSZ * SEQ_LEN)
    raw = token_ids[pos:pos + BSZ * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    x = torch.from_numpy(mapped).view(BSZ, SEQ_LEN).to(DEVICE)

    logits, h = model(x)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    consv = model.conservation_loss(x)
    loss = ce + CONSV_WEIGHT * consv

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    sched.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            pos = np.random.randint(0, N - 32 * SEQ_LEN)
            raw = token_ids[pos:pos + 32 * SEQ_LEN]
            mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
            tx = torch.from_numpy(mapped).view(32, SEQ_LEN).to(DEVICE)
            tl, th = model(tx)
            tce = F.cross_entropy(tl[:, :-1].reshape(-1, VOCAB), tx[:, 1:].reshape(-1))
            tppl = torch.exp(tce).item()
            tconsv = model.conservation_loss(tx).item()
            cs, cr = model.state_structure(tx)
            hf = th.view(-1, DM)
            # Decode first batch for a quick sanity check
            preds = tl[0].argmax(-1).cpu().numpy()
        model.train()
        pbar.set_postfix(
            ce=f'{ce.item():.2f}', cv=f'{consv.item():.4f}',
            ppl=f'{tppl:.1f}', tcv=f'{tconsv:.4f}',
            sr=f'{cs/max(cr,1e-8):.2f}',
            nrm=f'{hf.norm(dim=-1).mean().item():.2f}',
            lr=f'{sched.get_last_lr()[0]:.1e}',
        )
        torch.save({
            'model': model.state_dict(), 'opt': opt.state_dict(),
            'sched': sched.state_dict(), 'step': step,
        }, os.path.join(CKPT_DIR, f'info_consv_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min.')

# ═══════════════════════════════════════════════════
# Final Analysis
# ═══════════════════════════════════════════════════
print('\n=== Final Analysis ===')
model.eval()
with torch.no_grad():
    pos = np.random.randint(0, N - 64 * SEQ_LEN)
    raw = token_ids[pos:pos + 64 * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    tx = torch.from_numpy(mapped).view(64, SEQ_LEN).to(DEVICE)
    tl, th = model(tx)
    ce = F.cross_entropy(tl[:, :-1].reshape(-1, VOCAB), tx[:, 1:].reshape(-1))
    ppl = torch.exp(ce).item()
    consv = model.conservation_loss(tx).item()
    cs, cr = model.state_structure(tx)
    hf = th.view(-1, DM)

    print(f'  CE: {ce.item():.3f}  PPL: {ppl:.1f}')
    print(f'  Conservation loss: {consv:.4f}')
    print(f'  Self-cos: {cs:.4f}  Random-cos: {cr:.4f}  Ratio: {cs/max(cr,1e-8):.3f}')
    print(f'  State norm: mean={hf.norm(dim=-1).mean().item():.3f}')
    print(f'  Dim variance: {hf.var(dim=0).mean().item():.4f}')

    # Structure verification: are adjacent tokens' states more similar?
    first_half = th[:, :SEQ_LEN//2].reshape(-1, DM)
    second_half = th[:, SEQ_LEN//2:].reshape(-1, DM)
    n1 = F.normalize(first_half, dim=-1)
    n2 = F.normalize(second_half, dim=-1)
    within = (n1 @ n1.T).mean().item()
    cross = (n1 @ n2.T).mean().item()
    print(f'  Within-half cos: {within:.4f}  Cross-half cos: {cross:.4f}  Gap: {within-cross:.4f}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'info_consv_final.pt'))
print('Saved.')
