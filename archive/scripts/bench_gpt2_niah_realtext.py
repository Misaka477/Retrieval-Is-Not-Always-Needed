"""GPT-2 Real-text NIAH — same data as CANN/ABL, no slot, pure attention."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset

import torch, torch.nn.functional as F, random
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda"; torch.manual_seed(42)
random.seed(42)

V = 4096; DM = 416; N_LAYER = 6; N_HEAD = 8; MAX_SEQ = 512

# ── Data ──
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
paras = []
for t in ds["text"][:10000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 128:
            paras.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(paras)}", flush=True)

KEYS = list(range(1, 6))
VALS = list(range(6, 11))

def make_sample(gap):
    while True:
        p = paras[torch.randint(0, len(paras), (1,)).item()]
        need = 64 + gap + 4
        if len(p) >= need:
            seq = p[:need].tolist()
            k = random.choice(KEYS)
            v = random.choice(VALS)
            seq[0] = k; seq[1] = v; seq[-1] = k
            return torch.tensor(seq), v

def make_batch(n, gap):
    xs, ys = [], []
    for _ in range(n):
        x, v = make_sample(gap)
        xs.append(x); ys.append(v)
    return torch.stack(xs), torch.tensor(ys)

# ── Model (same 14.2M GPT-2) ──
cfg = GPT2Config(vocab_size=V, n_embd=DM, n_layer=N_LAYER, n_head=N_HEAD, n_positions=512)
model = GPT2LMHeadModel(cfg).to(device)
# Load pretrained LM weights for embed/head
st = torch.load("checkpoints/gpt2_15m_wt103_final.pt", map_location=device)
# Extend wpe from 64→512
wpe_old = st["transformer.wpe.weight"]
wpe_new = torch.zeros(512, DM, device=wpe_old.device)
wpe_new[:64] = wpe_old; wpe_new[64:] = wpe_old[-1:].repeat(448, 1)
st["transformer.wpe.weight"] = wpe_new
model.load_state_dict(st); model.eval()
print(f"  params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

@torch.no_grad()
def evaluate(x, y):
    out = model(x.to(device)).logits
    return (out[:, -1].argmax(-1) == y.to(device)).float().mean().item()

print("\n  gap    GPT-2 (recall)")
print("  ─────────────────────")
for gap in [8, 16, 32, 64, 128]:
    train_x, train_y = make_batch(200, gap)
    test_x, test_y = make_batch(200, gap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for step in range(200):
        model.train()
        model.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 32):
            idx = perm[i:i+32]
            logits = model(train_x[idx].to(device)).logits
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()

        if step % 10 == 9:
            model.eval()
            acc = evaluate(test_x, test_y)
            best = max(best, acc)
            print(f"  {gap:3d}  step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")

    print(f"  {gap:3d}  FINAL: {best*100:.0f}%")
    print()