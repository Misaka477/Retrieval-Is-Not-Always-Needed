"""RINA Phase 3a: LM training on P&P with progress bars."""
import torch, torch.nn.functional as F, sys, os, math, time
from tqdm import tqdm

sys.path.insert(0, "D:\\Software_Development\\Project\\RINA_Project\\references\\hopfield-layers")
from hflayers import Hopfield
from tokenizers import ByteLevelBPETokenizer

device = "cuda"
torch.manual_seed(42)

# ── Load tokenizer & data ──
tok = ByteLevelBPETokenizer.from_file(
    "D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\rina_4096-vocab.json",
    "D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\rina_4096-merges.txt")
V = tok.get_vocab_size()

pp = open("D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt",
          "r", encoding="latin-1").read()
ci = pp.find("CHAPTER")
pp = pp[ci:] if ci >= 0 else pp
data = torch.tensor(tok.encode(pp).ids, dtype=torch.long, device=device)
print(f"Data: {len(data):,} tokens  vocab={V}")

# ── Model ──
m = torch.nn.Sequential(
    torch.nn.Embedding(V, 256),
    Hopfield(256, 256, 256, num_heads=1, scaling=0.5,
             update_steps_max=3, batch_first=True),
    torch.nn.LayerNorm(256),
    torch.nn.Linear(256, V),
).to(device)
print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

# ── Train ──
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
SEQ, BS = 64, 16
EPOCHS = 50
t0 = time.time()
best_ppl = float("inf")

for ep in range(EPOCHS):
    m.train()
    total, nb = 0, 0
    perm = torch.randperm(len(data) - SEQ - 1, device=device)
    n_batches = (len(perm) + BS - 1) // BS

    pbar = tqdm(range(0, len(perm), BS), desc=f"ep {ep+1}/{EPOCHS}",
                unit="batch", leave=False)
    for i in pbar:
        idx = perm[i:i+BS]
        x = data[idx[:, None] + torch.arange(SEQ, device=device)]
        y = data[idx[:, None] + torch.arange(1, SEQ + 1, device=device)]
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        total += loss.item()
        nb += 1
        pbar.set_postfix(loss=f"{loss.item():.2f}")

    ppl = math.exp(total / nb)
    best_ppl = min(best_ppl, ppl)
    wall = (time.time() - t0) / 60
    tqdm.write(f"[ep={ep+1:2d}/{EPOCHS}] ppl={ppl:.2f}  best={best_ppl:.2f}  "
               f"loss={total/nb:.4f}  ({wall:.1f}min)")

    # Save checkpoint every 10 epochs
    if ep % 10 == 9:
        ckpt_path = f"D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\rina_lm_ep{ep+1}.pt"
        torch.save(m.state_dict(), ckpt_path)
        tqdm.write(f"  saved: {ckpt_path}")

    if ep % 10 == 9 or ep == EPOCHS - 1:
        m.eval()
        prompt = "The secret password is"
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            for _ in range(30):
                nid = m(x)[0, -1].argmax().item()
                x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
        gen = tok.decode(x[0].tolist())
        tqdm.write(f"  gen: {gen[:120]}")

torch.save(m.state_dict(),
           "D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\rina_lm_4096.pt")
print(f"\nDone ({(time.time()-t0)/60:.1f}min). Saved.")
