# -*- coding: utf-8 -*-
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "hopfield-layers"))

from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from hflayers import Hopfield

print("0: starting", flush=True)
device = "cuda"
torch.manual_seed(42)

V, DM = 4096, 512
SEQ, BS = 64, 16
EPOCHS, LR = 40, 3e-4
N_STORIES = 200000
BATCHES = 4000

print("1: loading data", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = []
for i, item in enumerate(ds):
    if i >= N_STORIES: break
    texts.append(item["text"])
print(f"2: loaded {len(texts):,} stories", flush=True)

print("3: training tokenizer", flush=True)
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(texts, vocab_size=V, min_frequency=2, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
V_actual = tok.get_vocab_size()
print(f"4: vocab={V_actual}", flush=True)

print("5: tokenizing", flush=True)
all_ids = []
for t in tqdm(texts, desc="tokenize"):
    all_ids.extend(tok.encode(t).ids)
data = torch.tensor(all_ids, dtype=torch.long, device=device)
print(f"6: {len(data):,} tokens, {data.element_size()*len(data)//1024**2}MB on GPU", flush=True)
del all_ids, texts

print("7: building model", flush=True)
m = torch.nn.Sequential(
    torch.nn.Embedding(V_actual, DM),
    Hopfield(DM, DM, DM, num_heads=2, scaling=0.5, update_steps_max=3, batch_first=True),
    torch.nn.LayerNorm(DM),
    Hopfield(DM, DM, DM, num_heads=2, scaling=0.5, update_steps_max=3, batch_first=True),
    torch.nn.LayerNorm(DM),
    torch.nn.Linear(DM, V_actual),
).to(device)
print(f"8: {sum(p.numel() for p in m.parameters()):,} params", flush=True)

print("9: starting training", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=LR)
t0 = time.time()
for ep in range(EPOCHS):
    m.train()
    tot, n = 0, 0
    for _ in tqdm(range(BATCHES), desc=f"ep{ep+1}/{EPOCHS}", leave=False):
        idx = torch.randint(0, len(data) - SEQ - 1, (BS,), device=device)
        x = data[idx[:,None] + torch.arange(SEQ, device=device)]
        y = data[idx[:,None] + torch.arange(1, SEQ + 1, device=device)]
        loss = F.cross_entropy(m(x).reshape(-1, V_actual), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        tot += loss.item(); n += 1
    ppl = math.exp(tot/n)
    et = (time.time()-t0)/60
    print(f"[{ep+1}/{EPOCHS}] ppl={ppl:.2f} loss={tot/n:.4f} ({et:.1f}min)", flush=True)
    if ep % 5 == 4 or ep == EPOCHS-1:
        torch.save(m.state_dict(), f"D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\ts_ep{ep+1}.pt")
        m.eval()
        prompt = "password "
        ids = tok.encode(prompt).ids
        xg = torch.tensor([ids], device=device)
        with torch.no_grad():
            for _ in range(40):
                nid = m(xg)[0, -1].argmax().item()
                xg = torch.cat([xg, torch.tensor([[nid]], device=device)], dim=1)
        print(f"  gen: {tok.decode(xg[0].tolist())[:120]}", flush=True)

torch.save(m.state_dict(), f"D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\ts_final.pt")
print(f"done ({time.time()-t0:.0f}s)", flush=True)
