# -*- coding: utf-8 -*-
"""RINA full: CANN-SSM + slot on Wikitext-2 real text."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.cann_ssm import RINASeqModel

print("0: starting", flush=True)
device = "cuda"
torch.manual_seed(42)
CKPT = "D:\\Software_Development\\Project\\RINA_Project\\checkpoints"
os.makedirs(CKPT, exist_ok=True)

V, DM = 4096, 256
SEQ, BS = 64, 8
EPOCHS, LR = 40, 3e-4
BATCHES = 2000

# ── 1. Load Wikitext-2 ──
print("1: loading Wikitext-2", flush=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = ds["text"]
print(f"2: {len(texts):,} lines", flush=True)

# ── 2. Train tokenizer ──
print("3: training tokenizer", flush=True)
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(texts, vocab_size=V, min_frequency=2,
                        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
tok.save_model(CKPT, "rina_wt")
V_actual = tok.get_vocab_size()
print(f"4: vocab={V_actual}", flush=True)

# ── 3. Tokenize ──
print("5: tokenizing", flush=True)
all_ids = []
for t in tqdm(texts, desc="tokenize"):
    all_ids.extend(tok.encode(t).ids)
data = torch.tensor(all_ids, dtype=torch.long, device=device)
print(f"6: {len(data):,} tokens, {data.element_size()*len(data)//1024**2}MB on GPU", flush=True)

# ── 4. Build model with slot ──
print("7: building RINA model", flush=True)
m = RINASeqModel(V_actual, d_model=DM, n_patterns=4096, beta=0.5,
                 attract_every=2, n_slots=4096).to(device)
print(f"8: {sum(p.numel() for p in m.parameters()):,} params", flush=True)

# ── 5. Train ──
print("9: training", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=LR)
t0 = time.time()

for ep in range(EPOCHS):
    m.train()
    tot, n = 0, 0
    for _ in tqdm(range(BATCHES), desc=f"ep{ep+1}/{EPOCHS}", leave=False):
        idx = torch.randint(0, len(data) - SEQ - 1, (BS,), device=device)
        x = data[idx[:, None] + torch.arange(SEQ, device=device)]
        y = data[idx[:, None] + torch.arange(1, SEQ + 1, device=device)]

        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, V_actual), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        n += 1

    ppl = math.exp(tot / n)
    et = (time.time() - t0) / 60
    print(f"[{ep+1}/{EPOCHS}] ppl={ppl:.2f} loss={tot/n:.4f} ({et:.1f}min)", flush=True)

    if ep % 5 == 4 or ep == EPOCHS - 1:
        torch.save(m.state_dict(), f"{CKPT}/rina_wt_ep{ep+1}.pt")
        m.eval()
        for prompt in ["The meaning of life is", "In the beginning", "Artificial intelligence"]:
            ids = tok.encode(prompt).ids
            xg = torch.tensor([ids], device=device)
            with torch.no_grad():
                for _ in range(40):
                    nid = m(xg)[0, -1].argmax().item()
                    if nid == 1: break
                    xg = torch.cat([xg, torch.tensor([[nid]], device=device)], dim=1)
            print(f"  [{prompt}] {tok.decode(xg[0].tolist())[:120]}", flush=True)

torch.save(m.state_dict(), f"{CKPT}/rina_wt_final.pt")
print(f"done ({(time.time()-t0)/60:.1f}min)", flush=True)
