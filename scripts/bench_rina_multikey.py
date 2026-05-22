"""RINA 15M — multi-key NIAH, matching bench_niah_multikey.py pattern."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from tokenizers import Tokenizer
import torch, torch.nn.functional as F, random
from tqdm import tqdm
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42); random.seed(42)
V = 4096; DM = 840; NP = 4096
KEYS = list(range(1, 6)); VALS = list(range(6, 11))
N_KEYS = 3; GAP = 128

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
paras = []
for t in ds["text"][:10000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 200:
            paras.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(paras)}", flush=True)

def make_batch(n):
    xs, ys, ks_list = [], [], []
    for _ in range(n):
        while True:
            p = paras[torch.randint(0, len(paras), (1,)).item()]
            need = 128 + GAP + N_KEYS * 3
            if len(p) >= need + 4:
                seq = p[:need].tolist()
                kvs = []
                for _ in range(N_KEYS):
                    k = random.choice(KEYS)
                    v = random.choice(VALS)
                    pos = random.randint(2, need - GAP - N_KEYS * 3)
                    seq[pos] = k
                    seq[pos + 1] = v
                    seq[-1 - (N_KEYS - len(kvs)) * 2] = k
                    kvs.append((pos, k, v))
                xs.append(torch.tensor(seq))
                ys.append([kv[2] for kv in kvs])
                ks_list.append(kvs)
                break
    return torch.stack(xs), torch.tensor(ys), ks_list

print("Generating batches...", flush=True)
train_x, train_y, train_kp = make_batch(200)
test_x, test_y, _ = make_batch(200)
print(f"  train: {train_x.shape}, test: {test_x.shape}", flush=True)

# Load model with slot
print("Loading RINA slot checkpoint...", flush=True)
sd = torch.load("checkpoints/cann_snn15m_v2_slot_ep13.pt", map_location=device, weights_only=False)
n = sum(p.numel() for p in TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5, n_slots=V).parameters())
m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                      attract_every=2, error_threshold=1.0,
                      hebbian_lr=0.0, inhibition_threshold=0.0,
                      n_slots=V).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
n = sum(p.numel() for p in m.parameters())
print(f"  params: {n/1e6:.1f}M", flush=True)

# Pre-populate slot table for training set
print("Populating slot table...", flush=True)
m.slot_table.zero_()
for i in range(len(train_x)):
    for _, k, v in train_kp[i]:
        m.slot_write(k, v)

# Fine-tune (positional loss only)
print("Fine-tuning (200 steps, positional loss)...", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
for step in tqdm(range(200), desc="RINA"):
    m.train(); m.zero_grad()
    for i in range(0, len(train_x), 16):
        idx = slice(i, min(i + 16, len(train_x)))
        out = m(train_x[idx].to(device))
        logits_q = out[:, -N_KEYS * 2::2]
        loss = F.cross_entropy(logits_q.reshape(-1, V), train_y[idx].view(-1).to(device))
        loss.backward()
    opt.step()
    if step % 10 == 9:
        m.eval()
        with torch.no_grad():
            o = m(test_x.to(device))
            a = (o[:, -N_KEYS * 2::2].argmax(-1) == test_y.to(device)).float().mean().item()
        best = max(best, a) if 'best' in dir() else a
        print(f"  step {step+1}: acc={a*100:.0f}% best={best*100:.0f}%", flush=True)
        m.train()

# Evaluate
print("Evaluating...", flush=True)
m.eval()
with torch.no_grad():
    out = m(test_x.to(device))
    preds = out[:, -N_KEYS * 2::2].argmax(-1)
    acc = (preds == test_y.to(device)).float().mean().item()

print(f"\n  RINA 15M (code-seq256, 200 steps): {acc*100:.0f}% multi-key ({N_KEYS}, gap={GAP})")
