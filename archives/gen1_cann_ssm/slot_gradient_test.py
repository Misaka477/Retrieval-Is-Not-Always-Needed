"""
Exp 1: Slot gradient path verification.
Mini RINA (dm=128, n_patterns=256), mixed LM + key-value training.
Goal: confirm slot_write gradient reaches embed/slot_proj.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch, torch.nn as nn, torch.nn.functional as F, random
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 128, 256
SEQ, BS = 64, 16
LR = 3e-4; EPOCHS = 3
SUBSAMPLE = 4; MAX_TOKENS = 50_000_000

# Small RINA from scratch (not loading any checkpoint)
from rina import TemporalSNNModel, TemporalSNNCell
from rina.drift import DriftTracker

# Build a minimal RINA with slot
model = TemporalSNNModel(VOCAB, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=2, error_threshold=0.5,
                          hebbian_lr=0.0, inhibition_threshold=0.0,
                          n_slots=VOCAB).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Mini RINA: {n/1e6:.2f}M params (dm={DM}, np={NP})")

# Override slot_write: REMOVE torch.no_grad() — this is the experiment
def slot_write_trainable(self, key_id, value_id):
    # key_id, value_id are Python ints
    k = torch.tensor([key_id], device=self.slot_table.device)
    v = torch.tensor([value_id], device=self.slot_table.device)
    val_emb = self.slot_proj(self.embed(v))
    # Use index_copy_ (differentiable) instead of direct assignment
    one_hot = F.one_hot(k, num_classes=self.slot_table.shape[0]).float().t()
    self.slot_table.data = self.slot_table + (one_hot @ val_emb - self.slot_table) * 0.5
    return val_emb  # return so caller can use it

# Monkey-patch
model.slot_write = lambda k, v: slot_write_trainable(model, k, v)

# Data
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:10000]

print("Tokenizing...")
all_ids = []
for t in tqdm(texts, desc="tokenizing"):
    ids = tok.encode(t).ids[:SEQ * 500]
    if len(ids) >= SEQ:
        all_ids.append(torch.tensor(ids, dtype=torch.long))
ids = torch.cat(all_ids) if all_ids else torch.zeros(0, dtype=torch.long)[:MAX_TOKENS]
ids = ids[:min(len(ids), MAX_TOKENS)]
print(f"  tokens: {len(ids):,}")

num_batches = (len(ids) - 1) // (BS * SEQ)
train_its = num_batches // SUBSAMPLE
print(f"  batches/epoch: {train_its}")

# NIAH data helper
KEYS = list(range(1, 21))
VALS = list(range(21, 41))

def make_niah_batch(bs, seq):
    x = torch.randint(VOCAB, (bs, seq))
    for b in range(bs):
        key = random.choice(KEYS)
        val = random.choice(VALS)
        kv_pos = random.randint(1, seq - 3)
        x[b, kv_pos] = key
        x[b, kv_pos + 1] = val
        x[b, -2] = key  # query at -2
        x[b, -1] = val  # target at -1
    return x

opt = torch.optim.AdamW(model.parameters(), lr=LR)
drift = DriftTracker(compute_coverage=True)

for ep in range(1, EPOCHS + 1):
    model.train(); total_loss = 0.0; niah_correct, niah_total = 0, 0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/{EPOCHS}")

    for bi in pbar:
        # 80% LM, 20% NIAH
        if bi % 5 == 0:
            x = make_niah_batch(BS, SEQ).to(device)
            for b in range(BS):
                k, v = int(x[b, 1]), int(x[b, 2])
                if k in KEYS and v in VALS:
                    model.slot_write(k, v)
            model.slot_table.data = model.slot_table.detach()
            is_niah = True
        else:
            start = perm[(bi * SUBSAMPLE) % len(perm)]
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            is_niah = False

        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        if is_niah:
            # Check if model predicted correctly at the last relevant position
            for b in range(BS):
                niah_total += 1
                if logits[b, -1].argmax().item() == int(x[b, -2]):
                    niah_correct += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

        if bi % 100 == 0 and bi > 0:
            ppl = torch.exp(torch.tensor(total_loss / bi)).item()
            slot_acc = 100 * niah_correct / max(niah_total, 1)
            pbar.set_postfix(ppl=f"{ppl:.1f}", slot=f"{slot_acc:.0f}%")

    ppl = torch.exp(torch.tensor(total_loss / train_its)).item()
    d = drift.step(model.cell.patterns)
    slot_acc = 100 * niah_correct / max(niah_total, 1)
    print(f"ep {ep}: ppl={ppl:.1f} slot={slot_acc:.1f}% "
          f"drift[cos={d['avg_cos']:.4f}, frob={d['frob_drift']:.4f}] "
          f"dead={d['dead_frac']*100:.0f}%")

print(f"\nDone. Slot acc: {slot_acc:.1f}% (random baseline: ~0.02%)")
print(f"PPL: {ppl:.1f}")
