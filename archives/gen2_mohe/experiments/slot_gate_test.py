"""
Exp 1: Slot trust gate — learn when to trust slot signal.
Mini RINA (dm=128, np=256), mixed LM + key-value data.
Goal: gate learns to trust slot at query positions.
Slot_write remains non-differentiable (buffer).
Slot_read_gate (Linear) is differentiable — learns when to use slot.
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

# Mini RINA with slot
from rina import TemporalSNNModel

model = TemporalSNNModel(VOCAB, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=2, error_threshold=0.5,
                          hebbian_lr=0.0, inhibition_threshold=0.0,
                          n_slots=VOCAB).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Mini RINA: {n/1e6:.2f}M params (dm={DM}, np={NP})")

# Add a learnable slot read gate: Linear(2*dm, 1) → sigmoid
# Decides "how much to trust slot signal" at each step
slot_read_gate = nn.Linear(DM * 2, 1).to(device)
slot_gate_opt = torch.optim.AdamW(slot_read_gate.parameters(), lr=LR)

# Monkey-patch forward to use slot_read_gate
orig_forward = model.forward
def gated_forward(self, x):
    bsz, seq_len = x.shape
    emb = self.embed(x)
    h = torch.zeros(bsz, DM, device=x.device)
    logits = []
    for t in range(seq_len):
        # Slot read with learned gate
        slot_val = self.slot_table[x[:, t]] if self.n_slots > 0 else 0
        gate_in = torch.cat([h, emb[:, t, :]], dim=-1)
        gate_val = torch.sigmoid(slot_read_gate(gate_in))
        h_in = h + gate_val * slot_val

        h = self.cell(h_in, emb[:, t, :], step=t)
        if t == seq_len - 1:
            pat = self.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h.unsqueeze(1)
            scores = xi @ pat.transpose(1, 2) * self.cell.beta_t[0]
            attn = torch.softmax(scores, dim=-1)
            attracted = (attn @ pat).squeeze(1)
            combined_last = torch.cat([h, emb[:, -1, :]], dim=-1)
            alpha = torch.sigmoid(self.cell.gate_alpha(combined_last))
            h = h + alpha * (attracted - h)
            h = self.cell.norm(h)
        logits.append(self.head(self.state_norm(h)))
    return torch.stack(logits, dim=1)

model.forward = lambda x: gated_forward(model, x)

# Data
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
print("Tokenizing WikiText...")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:10000]
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

# NIAH data
KEYS = list(range(1, 21)); VALS = list(range(21, 41))

def make_niah_batch(bs, seq):
    x = torch.randint(VOCAB, (bs, seq))
    for b in range(bs):
        key = random.choice(KEYS)
        val = random.choice(VALS)
        kv_pos = random.randint(1, seq - 3)
        x[b, kv_pos] = key
        x[b, kv_pos + 1] = val
        x[b, -2] = key
        x[b, -1] = val
    return x

opt = torch.optim.AdamW(model.parameters(), lr=LR)

for ep in range(1, EPOCHS + 1):
    model.train(); total_loss = 0.0; niah_correct, niah_total = 0, 0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/{EPOCHS}")

    for bi in pbar:
        if bi % 5 == 0:
            x = make_niah_batch(BS, SEQ).to(device)
            model.slot_table.zero_()
            for b in range(BS):
                k, v = int(x[b, 1]), int(x[b, 2])
                if k in KEYS and v in VALS:
                    model.slot_write(k, v)
            is_niah = True
        else:
            start = perm[(bi * SUBSAMPLE) % len(perm)]
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            is_niah = False

        opt.zero_grad(); slot_gate_opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))

        if is_niah:
            for b in range(BS):
                if int(x[b, 1]) in KEYS and int(x[b, 2]) in VALS:
                    niah_total += 1
                    if logits[b, -2].argmax(-1).item() == int(x[b, -1]):
                        niah_correct += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(slot_read_gate.parameters()), 1.0)
        opt.step(); slot_gate_opt.step()
        total_loss += loss.item()

        if bi % 100 == 0 and bi > 0:
            ppl = torch.exp(torch.tensor(total_loss / bi)).item()
            slot_acc = 100 * niah_correct / max(niah_total, 1)
            pbar.set_postfix(ppl=f"{ppl:.1f}", slot=f"{slot_acc:.0f}%")

    ppl = torch.exp(torch.tensor(total_loss / train_its)).item()
    slot_acc = 100 * niah_correct / max(niah_total, 1)
    print(f"ep {ep}: ppl={ppl:.1f} slot_acc={slot_acc:.1f}%")

print(f"\nDone. slot_acc={slot_acc:.1f}%")
print(f"Random baseline: ~0.02% (1/4096)")
print(f"Manual slot_write (no gate): ~22%")
print(f"Goal: > 60%")
