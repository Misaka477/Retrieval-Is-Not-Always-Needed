"""
Continued training on FineWeb (diverse web text, parquet format).
Replaces PG-19 (broken on datasets v4.x).
If ppl drops below WikiText-103 plateau (33.3), data ceiling not yet reached.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)

V, DM, NP, AE = 4096, 840, 4096, 2
SEQ, BS = 64, 8
LR = 1e-4; EPOCHS = 1
SUBSAMPLE = 8
MAX_TOKENS = 200_000_000
CKPT_SOURCE = "checkpoints/cann_snn15m_v2_slot_ep12.pt"

print(f"Config: dm={DM} seq={SEQ} bs={BS} lr={LR} ep={EPOCHS} max_tokens={MAX_TOKENS:,}")
print(f"Checkpoint: {CKPT_SOURCE}")

print("Loading checkpoint...", flush=True)
sd = torch.load(CKPT_SOURCE, map_location=device, weights_only=False)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.8,
                          n_slots=V).to(device)
model.load_state_dict(sd["model"], strict=False)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler()
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)

# Resume
start_ep = 1; resume_path = os.path.join(CKPT_DIR, "fineweb_resume.pt")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"]); start_ep = ckpt["ep"]
    print(f"  resume from ep {start_ep}", flush=True)

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

# FineWeb with streaming (parquet, no loading script needed)
print("Loading FineWeb sample-10BT...", flush=True)
bank = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
bank = bank.shuffle(buffer_size=10000, seed=42)

print("Tokenizing (limited to ~200M tokens)...", flush=True)
all_ids = []; total = 0
for sample in tqdm(bank, desc="tokenizing"):
    text = sample["text"]
    if len(text) < 200:
        continue
    ids = tok.encode(text).ids
    if len(ids) < SEQ:
        continue
    all_ids.append(ids[:min(len(ids), SEQ * 1000)])
    total += len(all_ids[-1])
    if total >= MAX_TOKENS:
        break

import numpy as np
ids = np.concatenate(all_ids)
ids = torch.tensor(ids, dtype=torch.long)
print(f"  tokens: {len(ids):,}", flush=True)

num_batches = (len(ids) - 1) // (BS * SEQ)
train_its = num_batches // SUBSAMPLE
print(f"  batches: {train_its}", flush=True)

for ep in range(start_ep, EPOCHS + 1):
    model.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/{EPOCHS}")
    for bi in pbar:
        if model.n_slots > 0:
            model.slot_table.zero_()
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
        if bi % 200 == 0 and bi > 0:
            pbar.set_postfix(ppl=f"{torch.exp(torch.tensor(total_loss/bi)):.1f}")

    ppl = torch.exp(torch.tensor(total_loss / train_its)).item()
    print(f"ep {ep}: ppl={ppl:.1f} on FineWeb")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "ep": ep, "ppl": ppl,
                "tokens": len(ids)},
               os.path.join(CKPT_DIR, f"fineweb_ep{ep}.pt"))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "ep": ep, "ppl": ppl},
               os.path.join(CKPT_DIR, "fineweb_resume.pt"))

print(f"Done. FineWeb ppl={ppl:.1f}")
if ppl < 33.3:
    print("Data ceiling NOT reached — scaling to more data is warranted.")
else:
    print("Further scaling experiments needed.")

