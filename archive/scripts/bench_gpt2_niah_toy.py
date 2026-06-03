"""GPT-2 Toy NIAH — same data as CANN/ABL, no slot, pure attention recall."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

import torch, torch.nn.functional as F, time, random
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda"; torch.manual_seed(42)

DM, N_KEYS = 64, 20
V = 2 * N_KEYS + 2  # 0=PAD, 1-20=keys, 21-40=values, 41=filler
SEQ = 150

def make_data(n, gap=8):
    f = V - 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, N_KEYS + 1, (1,)).item()
        v = torch.randint(N_KEYS + 1, 2 * N_KEYS + 1, (1,)).item()
        seq = [k, v] + [f] * gap + [k]
        x.append(seq)
        y.append(v)
    return torch.tensor(x), torch.tensor(y)

cfg = GPT2Config(vocab_size=V, n_embd=DM, n_layer=4, n_head=4, n_positions=SEQ)
model = GPT2LMHeadModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"GPT-2 NIAH model: {n:,} params ({n/1e6:.1f}M)", flush=True)

@torch.no_grad()
def evaluate(x, y):
    out = model(x.to(device)).logits
    pred = out[:, -1].argmax(-1)
    return (pred == y.to(device)).float().mean().item()

print("\n  gap    GPT-2 (recall)")
print("  ─────────────────────")
for gap in [8, 16, 32, 64, 128]:
    train_x, train_y = make_data(400, gap)
    test_x, test_y = make_data(200, gap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for step in range(200):
        model.zero_grad()
        logits = model(train_x.to(device)).logits
        loss = F.cross_entropy(logits[:, -1], train_y.to(device))
        loss.backward(); opt.step()

        if step % 10 == 9:
            model.eval()
            acc = evaluate(test_x, test_y)
            best = max(best, acc)
            print(f"  {gap:3d}  step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")
            model.train()

    print(f"  {gap:3d}  FINAL: {best*100:.0f}%")
    print()