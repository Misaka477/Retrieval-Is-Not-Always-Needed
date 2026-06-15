"""World Model Phase 2: Train language head + text generation evaluation."""
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# ═══════════════════════════════════════════════════
# Same WorldModel architecture
# ═══════════════════════════════════════════════════

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DM)
        self.transition = nn.Sequential(
            nn.Linear(DM * 2, DM), nn.GELU(), nn.Linear(DM, DM),
        )
        self.predict_obs = nn.Linear(DM, DM)
        self.state_init = nn.Parameter(torch.zeros(DM))
        for m in [self.transition, self.predict_obs]:
            for p in m.parameters():
                if p.dim() >= 2: nn.init.xavier_uniform_(p, 0.5)

    def forward(self, x):
        B, T = x.shape
        emb = self.embed(x)
        state = self.state_init.unsqueeze(0).expand(B, -1)
        states, losses = [state], []
        for t in range(T):
            inp = torch.cat([state, emb[:, t]], dim=-1)
            state = self.transition(inp)
            states.append(state)
        states = torch.stack(states, dim=1)
        preds = self.predict_obs(states[:, :-1])
        loss = F.mse_loss(preds, emb)
        with torch.no_grad():
            cos = F.cosine_similarity(preds.view(-1, DM), emb.view(-1, DM)).mean()
        return states, loss, cos

    @torch.no_grad()
    def generate(self, prompt, head, steps=50, temp=0.8):
        """Generate text from the world model + language head."""
        self.eval()
        head.eval()
        g = prompt.clone()
        state = self.state_init.unsqueeze(0).expand(1, -1)
        for _ in range(steps):
            emb = self.embed(g[:, -1:])
            inp = torch.cat([state, emb], dim=-1)
            state = self.transition(inp)
            logits = head(state)
            probs = torch.softmax(logits[:, -1] / temp, -1)
            next_tok = torch.multinomial(probs, 1)
            g = torch.cat([g, next_tok], dim=1)
        return g


class LanguageHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(DM, VOCAB, bias=False)
        nn.init.xavier_uniform_(self.head.weight, 0.5)

    def forward(self, state):
        return self.head(state)


# ═══════════════════════════════════════════════════
# Load trained world model + data
# ═══════════════════════════════════════════════════

wm = WorldModel().to(DEVICE)
latest = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith('worldmodel_') and f.endswith('.pt') and 'final' not in f])[-1]
ckpt = torch.load(os.path.join(CKPT_DIR, latest), weights_only=False)
print(f'Loaded {latest}')
wm.load_state_dict(ckpt['wm'])
print(f'Loaded world model from step {ckpt["step"]}')

token_ids = np.load(os.path.join(CKPT_DIR, 'mohe_fw_rwkv_1b.npy'), mmap_mode='r')
N = token_ids.shape[0]
from collections import Counter
sample_ids = token_ids[np.random.default_rng(42).integers(0, N, size=50000)]
freq = Counter(sample_ids.tolist())
top_vocab = [t for t, _ in freq.most_common(VOCAB)]
token_to_id = {t: i for i, t in enumerate(top_vocab)}
oov = VOCAB - 1

# Simple tokenizer wrapper (no Ninja needed)
TOKENIZER_PATH = os.path.join(CKPT_DIR, 'rwkv_vocab_v20230424.txt')
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, encoding='utf-8') as f:
        _vocab = {i: line.split(' ')[0] for i, line in enumerate(f)
                  if line.strip()}
    _max_id = max(_vocab.keys()) if _vocab else 0

    def simple_encode(text):
        return [ord(c) % VOCAB for c in text[:50]] or [0]

    def simple_decode(ids):
        return ''.join(chr(i % 128) if 32 <= (i % 128) < 127 else '?' for i in ids)

    has_tokenizer = True
    encode_fn, decode_fn = simple_encode, simple_decode
    print(f'Simple tokenizer loaded ({len(_vocab)} tokens)')
else:
    has_tokenizer = False
    encode_fn = decode_fn = None
    print('No tokenizer file found')

# ═══════════════════════════════════════════════════
# Phase 2: Train language head
# ═══════════════════════════════════════════════════

print('\n=== Phase 2: Language Head ===')
wm.eval()
lm = LanguageHead().to(DEVICE)
print(f'Head params: {sum(p.numel() for p in lm.parameters())/1e3:.1f}K')

opt = torch.optim.AdamW(lm.parameters(), lr=1e-3)
N_STEPS = 5000
pbar = tqdm(range(N_STEPS))
for step in pbar:
    pos = np.random.randint(0, N - BSZ * SEQ_LEN)
    raw = token_ids[pos:pos + BSZ * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    x = torch.from_numpy(mapped).view(BSZ, SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        states, _, _ = wm(x)
        states = states[:, :-1]
    logits = lm(states)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
    opt.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(lm.parameters(), 5.0)
    opt.step()
    if step % 1000 == 0:
        pbar.set_postfix(ce=f'{ce.item():.3f}', ppl=f'{torch.exp(ce).item():.0f}')

print(f'Final: CE={ce.item():.3f} PPL={torch.exp(ce).item():.0f}')

# ═══════════════════════════════════════════════════
# Evaluate: generation + state analysis
# ═══════════════════════════════════════════════════

print('\n=== Evaluation ===')
lm.eval()
with torch.no_grad():
    pos = np.random.randint(0, N - 64 * SEQ_LEN)
    raw = token_ids[pos:pos + 64 * SEQ_LEN]
    mapped = np.array([token_to_id.get(int(t), oov) for t in raw], dtype=np.int64)
    tx = torch.from_numpy(mapped).view(64, SEQ_LEN).to(DEVICE)

    states, wm_loss, wm_cos = wm(tx)
    s_align = states[:, :-1]
    logits = lm(s_align)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), tx.reshape(-1))
    ppl = torch.exp(ce).item()

    # State structure
    hf = s_align.reshape(-1, DM)
    nrm = F.normalize(hf, dim=-1)
    idx = torch.randperm(hf.size(0), device=DEVICE)
    sim_s = (nrm @ nrm.T).mean().item()
    sim_r = (nrm @ nrm[idx].T).mean().item()

    print(f'  World model: loss={wm_loss.item():.4f} cos={wm_cos.item():.3f}')
    print(f'  Language head: CE={ce.item():.3f} PPL={ppl:.1f}')
    print(f'  State structure ratio: {sim_s/max(sim_r,1e-8):.3f}')
    print(f'  State norm: {hf.norm(dim=-1).mean().item():.2f}')

    # ════════════════════════════════════════════
    # Text generation
    # ════════════════════════════════════════════
    print('\n--- Text Generation ---')
    prompts_raw = [
        "The capital of France is",
        "The Eiffel tower is in",
        "User: What is 2+2?\n\nAssistant:",
        "Once upon a time, there was",
    ]
    for prompt in prompts_raw:
        print(f'\nPrompt: {prompt}')
        try:
            ids = encode_fn(prompt) if has_tokenizer else [4]*8
            p = torch.tensor([ids], device=DEVICE)
            g = wm.generate(p, lm, steps=40, temp=0.8)
            if has_tokenizer:
                decoded = decode_fn(g[0].tolist()[len(ids):])
                print(f'  Generated: {repr(decoded[:100])}')
            else:
                print(f'  IDs: {g[0].tolist()[:20]}')
        except Exception as e:
            print(f'  Generation failed: {e}')

torch.save({'wm': wm.state_dict(), 'head': lm.state_dict()},
           os.path.join(CKPT_DIR, 'worldmodel_final.pt'))
print('\nDone.')
