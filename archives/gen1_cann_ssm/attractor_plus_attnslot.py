"""
Exp 3: Attractor layer + Attention Slot — combined.
Architecture:
  SSM Gate → ε-gated Attractor → semantic state h
  Attention Slot Memory → content-addressable retrieval → h += slot_read
  
Tests: LM ppl + slot recall accuracy.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_DATASETS_OFFLINE"] = "1"

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn as nn, torch.nn.functional as F, random
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 128, 256
SEQ, BS = 64, 8
LR = 1e-3; STEPS = 3000
SUBSAMPLE = 4; MAX_TOKENS = 50_000_000
N_SLOTS = 256
NIAH_EVERY = 5  # every 5th batch is NIAH

print(f"Exp 3: Attractor + Attention Slot (dm={DM}, np={NP}, slots={N_SLOTS})")

# ── Components ──
from rina.cell import TemporalSNNCell
cell = TemporalSNNCell(DM, NP, error_threshold=0.5, hebbian_lr=0.0).to(device)
embed = nn.Embedding(VOCAB, DM).to(device)
head = nn.Linear(DM, VOCAB).to(device)
state_norm = nn.LayerNorm(DM).to(device)

# Attention Slot Memory
class AttnSlot(nn.Module):
    def __init__(self, d_model, n_slots, beta=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.beta = beta
        # Learnable key-value memory bank
        self.key_bank = nn.Parameter(torch.randn(n_slots, d_model) * 0.01)
        self.val_bank = nn.Parameter(torch.randn(n_slots, d_model) * 0.01)
        # Write gate: decides whether to store current (h, x) pair
        self.write_gate = nn.Linear(DM * 2, 1)
        self.usage = nn.Parameter(torch.zeros(n_slots))  # learnable slot replacement

    def read(self, query):
        scores = query @ self.key_bank.T * self.beta
        attn = F.softmax(scores, dim=-1)
        return attn @ self.val_bank

    def write(self, key_vec, val_vec):
        # Content-based write: replace least-used slot
        with torch.no_grad():
            scores = key_vec @ self.key_bank.T * self.beta
            least_used = scores.argmin(dim=-1)  # slot with least similar key
        for i in range(key_vec.shape[0]):
            sidx = least_used[i].item()
            self.key_bank.data[sidx] = key_vec[i].detach()
            self.val_bank.data[sidx] = val_vec[i].detach()

slot_mem = AttnSlot(DM, N_SLOTS).to(device)

params = list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()) + \
         list(state_norm.parameters()) + list(slot_mem.parameters())
n = sum(p.numel() for p in params)
print(f"  Total params: {n/1e6:.2f}M")
print(f"  Cell: {sum(p.numel() for p in cell.parameters())/1e3:.0f}K")
print(f"  Slot memory: {sum(p.numel() for p in slot_mem.parameters())/1e3:.0f}K")

opt = torch.optim.AdamW(params, lr=LR)

# ── Data ──
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

# NIAH data
def make_niah_batch(bs, seq):
    x = torch.randint(VOCAB, (bs, seq))
    keys = [random.randint(2, VOCAB - 1) for _ in range(bs)]
    vals = [random.randint(2, VOCAB - 1) for _ in range(bs)]
    for b in range(bs):
        while vals[b] == keys[b]: vals[b] = random.randint(2, VOCAB - 1)
        kv_pos = random.randint(1, seq - 3)
        x[b, kv_pos] = keys[b]
        x[b, kv_pos + 1] = vals[b]
        x[b, -2] = keys[b]
        x[b, -1] = vals[b]
    return x, keys, vals

# ── Training ──
slot_recall_correct, slot_recall_total = 0, 0
tracker = {"correct": 0, "total": 0}

for step in range(STEPS):
    if step % NIAH_EVERY == 0:
        x, keys, vals = make_niah_batch(BS, SEQ)
        x = x.to(device)
        # Write key→value to slot memory
        k_emb = embed(torch.tensor(keys, device=device))
        v_emb = embed(torch.tensor(vals, device=device))
        slot_mem.write(k_emb, v_emb)
        is_niah = True
    else:
        start = random.randint(0, len(ids) - BS * SEQ - 1)
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        is_niah = False

    # Forward
    emb = embed(x)
    h = torch.zeros(BS, DM, device=device)
    logits = []
    for t in range(SEQ):
        # 1) Slot read (content-addressable)
        slot_out = slot_mem.read(h)  # query by current state
        read_gate = torch.sigmoid((h * slot_out).sum(dim=-1, keepdim=True))
        h = h + read_gate * slot_out * 0.3

        # 2) SSM gate + attractor
        h = cell(h, emb[:, t, :], step=t)
        logits.append(head(state_norm(h)))
    logits = torch.stack(logits, dim=1)

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    # Track slot recall
    if is_niah:
        with torch.no_grad():
            preds = logits[:, -2].argmax(-1)
            for b in range(BS):
                tracker["total"] += 1
                if preds[b].item() == vals[b]:
                    tracker["correct"] += 1

    if step % 200 == 199:
        slot_acc = 100 * tracker["correct"] / max(tracker["total"], 1)
        print(f"  step {step+1}: ppl={torch.exp(loss).item():.1f} slot_acc={slot_acc:.0f}%")

slot_acc = 100 * tracker["correct"] / max(tracker["total"], 1)
ppl_final = torch.exp(loss).item()
print(f"\n=== Results ===")
print(f"  Final ppl:     {ppl_final:.1f}")
print(f"  Slot recall:   {slot_acc:.1f}% (random: ~0.02%)")
print(f"  Architecture:  attractor dm={DM} + attention slot N={N_SLOTS}")
print(f"  Training:      {STEPS} steps, {NIAH_EVERY}x NIAH mixing")
