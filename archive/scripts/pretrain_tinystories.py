# -*- coding: utf-8 -*-
"""RINA Phase 3a: Train LM on TinyStories."""
import torch
import torch.nn.functional as F
import sys, os, math, time
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, "D:\\Software_Development\\Project\\RINA_Project\\references\\hopfield-layers")
from hflayers import Hopfield
from tokenizers import ByteLevelBPETokenizer

device = "cuda"
torch.manual_seed(42)
CKPT = "D:\\Software_Development\\Project\\RINA_Project\\checkpoints"
os.makedirs(CKPT, exist_ok=True)

V, DM = 4096, 256
SEQ, BS = 64, 16
EPOCHS, LR = 30, 3e-4
N_STORIES = 200000

print("[1/5] Loading TinyStories...", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = []
for i, item in enumerate(ds):
    if i >= N_STORIES:
        break
    texts.append(item["text"])
print(f"Loaded {len(texts):,} stories")

print("[2/5] Training BPE tokenizer...")
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(texts, vocab_size=V, min_frequency=2,
                        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
tok.save_model(CKPT, "ts_4096")
V_actual = tok.get_vocab_size()
print(f"Vocab: {V_actual}")

print("[3/5] Tokenizing...")
all_ids = []
for t in tqdm(texts, desc="Tokenize"):
    all_ids.extend(tok.encode(t).ids)
data = torch.tensor(all_ids, dtype=torch.long, device="cuda")
print(f"Tokens: {len(data):,} | GPU: {data.element_size()*len(data)//1024**2}MB")

print("[4/5] Building model...")
m = torch.nn.Sequential(
    torch.nn.Embedding(V_actual, DM),
    Hopfield(DM, DM, DM, num_heads=1, scaling=0.5,
             update_steps_max=3, batch_first=True),
    torch.nn.LayerNorm(DM),
    torch.nn.Linear(DM, V_actual),
).to(device)
print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

print("[5/5] Training...")
opt = torch.optim.AdamW(m.parameters(), lr=LR)
t0 = time.time()

for ep in range(EPOCHS):
    m.train()
    total, nb = 0, 0
    perm = torch.randperm(len(data) - SEQ - 1, device=device)

    for i in tqdm(range(0, len(perm), BS), desc=f"ep {ep+1}/{EPOCHS}", unit="batch", leave=False):
        idx = perm[i:i+BS]
        x = data[idx[:, None] + torch.arange(SEQ, device=device)]
        y = data[idx[:, None] + torch.arange(1, SEQ + 1, device=device)]
        loss = F.cross_entropy(m(x).reshape(-1, V_actual), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        total += loss.item()
        nb += 1

    ppl = math.exp(total / nb)
    tqdm.write(f"[ep={ep+1:2d}/{EPOCHS}] ppl={ppl:.2f} loss={total/nb:.4f} ({time.time()-t0:.0f}s)")

    if ep % 5 == 4 or ep == EPOCHS - 1:
        torch.save(m.state_dict(), f"{CKPT}/tots_ep{ep+1}.pt")
        m.eval()
        prompt = "The secret password is"
        ids = tok.encode(prompt).ids
        xg = torch.tensor([ids], device=device)
        with torch.no_grad():
            for _ in range(40):
                nid = m(xg)[0, -1].argmax().item()
                xg = torch.cat([xg, torch.tensor([[nid]], device=device)], dim=1)
        tqdm.write(f"  gen: {tok.decode(xg[0].tolist())[:120]}")

torch.save(m.state_dict(), f"{CKPT}/tots_final.pt")
print(f"Done ({(time.time()-t0)/60:.1f}min). Saved.")
