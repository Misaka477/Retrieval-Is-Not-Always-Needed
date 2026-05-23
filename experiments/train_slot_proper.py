"""
Proper slot training: differentiable READ via nn.Embedding slot_table.
No slot_table.zero_(). Slot_write via .data (non-diff),
slot_read via Embedding lookup (diff — gradient flows through).
Train on mixed LM+NIAH, long seq, 3.7M RINA cell.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn as nn, torch.nn.functional as F, random
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 256, 1024
SEQ, BS = 256, 4
LR = 3e-4; STEPS = 1000
NIAH_EVERY = 5

from rina.cell import TemporalSNNCell
cell = TemporalSNNCell(DM, NP, error_threshold=1.0, hebbian_lr=0.0).to(device)
embed = nn.Embedding(VOCAB, DM).to(device)
head = nn.Linear(DM, VOCAB).to(device)
state_norm = nn.LayerNorm(DM).to(device)

# Differentiable slot — read path gradients ARE differentiable
slot_table = nn.Embedding(VOCAB, DM).to(device)
nn.init.normal_(slot_table.weight, mean=0.0, std=0.01)
slot_read_gate = nn.Linear(DM * 2, 1).to(device)
slot_proj = nn.Linear(DM, DM).to(device)

params = list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()) + \
         list(state_norm.parameters()) + list(slot_table.parameters()) + list(slot_read_gate.parameters()) + list(slot_proj.parameters())
n = sum(p.numel() for p in params)
print(f"Model: {n/1e6:.2f}M params (dm={DM}, np={NP})")
opt = torch.optim.AdamW(params, lr=LR)

# ── Data ──
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
print("Loading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:5000]
ids_list = []
for t in tqdm(texts, desc="tokenizing"):
    e = tok.encode(t).ids[:SEQ * 50]
    if len(e) >= SEQ:
        ids_list.append(torch.tensor(e[:SEQ * 10], dtype=torch.long))
ids = torch.cat(ids_list)[:200000] if ids_list else torch.zeros(0, dtype=torch.long)
nb = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  {len(ids):,} tokens, {nb} batches/epoch")

def make_niah(bs, seq):
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
print("Training (mixed LM + slot-aware)...")
slot_correct, slot_total = 0, 0

for step in range(STEPS):
    if step % NIAH_EVERY == 0:
        x, keys, vals = make_niah(BS, SEQ)
        x = x.to(device)
        # Write key→value TO THE DIFFERENTIABLE slot_table
        for b in range(BS):
            k_onehot = F.one_hot(torch.tensor([keys[b]], device=device), VOCAB).float().t()
            v_emb = slot_proj(embed(torch.tensor([vals[b]], device=device)))
            # Differentiable write: slot_table.weight[keys[b]] = v_emb
            # Use a soft write: slot_table.weight += one_hot * (v_emb - slot_table.weight) * 0.5
            slot_table.weight.data[keys[b]] = v_emb.squeeze(0).detach()
    else:
        bi = step % nb
        x = ids[bi * BS * SEQ : (bi + 1) * BS * SEQ].view(BS, SEQ).to(device)

    # Forward with RINA cell + slot read
    emb = embed(x)
    h = torch.zeros(BS, DM, device=device)
    logits = []
    for t in range(SEQ):
        # Slot read (differentiable — nn.Embedding lookup)
        slot_val = slot_table(x[:, t])
        gate_in = torch.cat([h, emb[:, t, :]], dim=-1)
        gate_val = torch.sigmoid(slot_read_gate(gate_in))
        h = cell(h + gate_val * slot_val * 0.1, emb[:, t, :], step=t)
        logits.append(head(state_norm(h)))
    logits = torch.stack(logits, dim=1)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))

    # Track slot accuracy on NIAH batches
    if step % NIAH_EVERY == 0:
        preds = logits[:, -2].argmax(-1)
        for b in range(BS):
            slot_total += 1
            if preds[b].item() == vals[b]:
                slot_correct += 1

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    if step % 100 == 99:
        ppl = torch.exp(loss).item()
        sa = 100 * slot_correct / max(slot_total, 1)
        print(f"  step {step+1}: ppl={ppl:.1f} slot_acc={sa:.1f}%")

sa = 100 * slot_correct / max(slot_total, 1)
print(f"\nDone. Final slot_acc={sa:.1f}% (random: 0.024%)")
print(f"Slot {'learning' if sa > 1 else 'NOT learning'} — {'TRAINING WORKS' if sa > 1 else 'NEED DIFFERENT APPROACH'}")
