"""World Model with Contrastive Loss — semantic structure in state space.

Key difference from MSE version:
- Instead of predicting exact embedding values (MSE),
  we train the state to be cos-similar to the true next embedding
  and dissimilar to random embeddings.

Contrastive loss naturally clusters semantically similar positions
together in state space, because tokens that appear in similar contexts
will have similar embeddings → model pulls them together.
"""
import os, sys, time, glob
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

print(f'Contrastive World Model: DM={DM} V={VOCAB} SEQ={SEQ_LEN} BS={BSZ}')

# ═══════════════════════════════════════════════════
# World Model (same architecture, different loss)
# ═══════════════════════════════════════════════════

class WorldModel(nn.Module):
    def __init__(self, n_transition_layers=4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DM)
        
        # Deeper transition: stack multiple blocks
        layers = []
        in_dim = DM * 2
        for i in range(n_transition_layers):
            out_dim = DM if i == n_transition_layers - 1 else DM * 2
            layers.extend([
                nn.Linear(in_dim, DM * 2),
                nn.GELU(),
            ])
            in_dim = DM * 2
        layers.append(nn.Linear(DM * 2, DM))
        self.transition = nn.Sequential(*layers)
        
        self.n_layers = n_transition_layers
        self.predict_obs = nn.Linear(DM, DM)
        self.state_init = nn.Parameter(torch.zeros(DM))
        for m in [self.transition, self.predict_obs]:
            for p in m.parameters():
                if p.dim() >= 2: nn.init.xavier_uniform_(p, 0.5)

    def forward(self, x):
        B, T = x.shape
        emb = self.embed(x)
        state = self.state_init.unsqueeze(0).expand(B, -1)
        states = [state]
        for t in range(T):
            inp = torch.cat([state, emb[:, t]], dim=-1)
            state = self.transition(inp)
            states.append(state)
        states = torch.stack(states, dim=1)
        pred_obs = self.predict_obs(states[:, :-1])
        return states, pred_obs, emb

    @torch.no_grad()
    def state_structure(self, x):
        B, T = x.shape
        states, _, _ = self.forward(x)
        hf = states[:, :-1].reshape(-1, DM)
        normed = F.normalize(hf, dim=-1)
        sim_self = (normed @ normed.T).mean().item()
        idx = torch.randperm(hf.size(0), device=DEVICE)
        sim_shuf = (normed @ normed[idx].T).mean().item()
        return sim_self, sim_shuf


def contrastive_loss(pred_obs, target_emb, tau=0.5):
    """InfoNCE loss with in-batch negatives.
    
    For each (pred_i, target_i), positives are cos(pred_i, target_i).
    Negatives are cos(pred_i, target_j) for all j != i.
    """
    pn = F.normalize(pred_obs, dim=-1)  # [N, D]
    tn = F.normalize(target_emb, dim=-1)  # [N, D]
    
    # All pairwise similarities: [N, N]
    logits = (pn @ tn.T) / tau
    # Diagonal = positive pairs
    labels = torch.arange(pn.size(0), device=DEVICE)
    
    loss = F.cross_entropy(logits, labels)
    
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
    
    return loss, acc


# ═══════════════════════════════════════════════════
# Data & Training
# ═══════════════════════════════════════════════════

token_ids = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = token_ids.shape[0]

from collections import Counter
rng = np.random.default_rng(42)
sample_ids = token_ids[rng.integers(0, N, size=50000)]
freq = Counter(sample_ids.tolist())
top_vocab = [t for t, _ in freq.most_common(VOCAB)]
token_to_id = {t: i for i, t in enumerate(top_vocab)}
oov = VOCAB - 1
print(f'Vocab: {len(top_vocab)}/{VOCAB}, cov: {sum(freq[t] for t in top_vocab)/sum(freq.values())*100:.1f}%')

model = WorldModel().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f'Params: {params/1e6:.2f}M')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)

model.train()
pbar = tqdm(range(N_STEPS))
step = 0
t0 = time.time()
history = []

for step in pbar:
    pos = np.random.randint(0, N - BSZ * SEQ_LEN)
    raw = token_ids[pos:pos + BSZ * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    x = torch.from_numpy(mapped).view(BSZ, SEQ_LEN).to(DEVICE)

    states, pred_obs, emb = model(x)
    # pred_obs: [B, T, DM] — predict next obs from state_before
    # emb: [B, T, DM] — actual observation
    pred = pred_obs.reshape(-1, DM)
    target = emb.detach().reshape(-1, DM)

    loss, acc = contrastive_loss(pred, target)

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
            ts, tp, te = model(tx)
            tloss, tacc = contrastive_loss(tp.reshape(-1, DM), te.reshape(-1, DM))
            cs, cr = model.state_structure(tx)
            hf = ts[:, :-1].reshape(-1, DM)
        model.train()
        history.append((step, loss.item(), acc, cs, cr, hf.norm(dim=-1).mean().item()))
        pbar.set_postfix(
            loss=f'{loss.item():.4f}', acc=f'{acc:.3f}',
            tloss=f'{tloss.item():.4f}', tacc=f'{tacc:.3f}',
            sr=f'{cs/max(cr,1e-8):.2f}',
            lr=f'{sched.get_last_lr()[0]:.1e}',
        )
        torch.save({
            'model': model.state_dict(), 'opt': opt.state_dict(),
            'sched': sched.state_dict(), 'step': step,
        }, os.path.join(CKPT_DIR, f'contrast_wm_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min.')

# ═══════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════
print('\n=== Analysis ===')
model.eval()
with torch.no_grad():
    pos = np.random.randint(0, N - 64 * SEQ_LEN)
    raw = token_ids[pos:pos + 64 * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    tx = torch.from_numpy(mapped).view(64, SEQ_LEN).to(DEVICE)

    states, pred_obs, emb = model(tx)
    loss, acc = contrastive_loss(pred_obs.reshape(-1, DM), emb.reshape(-1, DM))
    cs, cr = model.state_structure(tx)
    hf = states[:, :-1].reshape(-1, DM)

    print(f'  Contrastive loss: {loss.item():.4f}  acc: {acc:.3f}')
    print(f'  State self-cos: {cs:.4f}  Random-cos: {cr:.4f}  Ratio: {cs/max(cr,1e-8):.3f}')
    print(f'  State norm: {hf.norm(dim=-1).mean().item():.2f}')
    print(f'  Dim variance: {hf.var(dim=0).mean().item():.4f}')

    # Positional structure
    print(f'\n  Positional structure (state at t vs state at t+k):')
    sa = states[:, :-1]
    for gap in [0, 1, 4, 16, 32]:
        if gap < sa.size(1):
            sim = F.cosine_similarity(sa[:, 0], sa[:, gap]).mean().item()
            print(f'    gap={gap:>2d}: cos = {sim:.4f}')

    # Training trajectory
    print(f'\n  Training trajectory (step, loss, acc, ratio):')
    for h in history[::4]:
        print(f'    step {h[0]:>5d}: loss={h[1]:.4f} acc={h[2]:.3f} ratio={h[4]/max(h[5],1e-8):.2f}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'contrast_wm_final.pt'))
print('\nSaved.')
