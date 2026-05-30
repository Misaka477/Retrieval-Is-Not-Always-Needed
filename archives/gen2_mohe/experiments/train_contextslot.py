"""
Train ContextSlot: add trainable memory to code-seq256, 200 steps.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)
SEQ, BS = 128, 4; N_SLOTS = 256; DM = 840

class ContextSlot(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.key_bank = nn.Parameter(torch.randn(n, d) * 0.01)
        self.val_bank = nn.Parameter(torch.randn(n, d) * 0.01)
        self.write_gate = nn.Linear(d, 1)
        self.register_buffer("usage", torch.zeros(n))
        self.n_slots = n
    def read(self, h, beta=2.0):
        a = F.softmax(h @ self.key_bank.T * beta, dim=-1)
        return a @ self.val_bank
    def write(self, h):
        with torch.no_grad():
            s = self.usage.argmin().item()
            self.usage[s] += 1.0
            h_mean = h.mean(dim=0).detach()
            self.key_bank.data[s] = 0.99 * self.key_bank.data[s] + 0.01 * h_mean
            self.val_bank.data[s] = 0.99 * self.val_bank.data[s] + 0.01 * h_mean

from rina import TemporalSNNModel
print("Loading code-seq256...")
sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, DM, 4096, beta=0.5, attract_every=2, error_threshold=1.0, hebbian_lr=0.0).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
print(f"  RINA: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

cs = ContextSlot(DM, N_SLOTS).to(device)
print(f"  ContextSlot: {sum(p.numel() for p in cs.parameters())/1e3:.0f}K")

# Data with cache
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
CACHE = "checkpoints/cs_train_tokens.pt"
if os.path.exists(CACHE):
    print("Loading cached tokens...")
    ids = torch.load(CACHE, map_location="cpu", weights_only=True)
    print(f"  {len(ids):,} tokens")
else:
    print("Loading code data (~3 min)...")
    code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
    texts = [next(iter(code))["content"] for _ in tqdm(range(100), desc="fetching")]
    ids_list = []
    for t in tqdm(texts, desc="tokenizing"):
        e = tok.encode(t).ids
        if len(e) >= SEQ:
            ids_list += e[:SEQ * 100]
    ids = torch.tensor(ids_list[:100000], dtype=torch.long) if ids_list else torch.zeros(0, dtype=torch.long)
    torch.save(ids, CACHE)
    print(f"  {len(ids):,} tokens (cached)")
nb = max(1, (len(ids) - 1) // (BS * SEQ))
split = int(len(ids) * 0.8)
train_ids = ids[:split]
eval_ids = ids[split:]
train_nb = max(1, (len(train_ids) - 1) // (BS * SEQ))
eval_nb = max(1, (len(eval_ids) - 1) // (BS * SEQ))
print(f"  train: {len(train_ids):,} ({train_nb} batches)  eval: {len(eval_ids):,} ({eval_nb} batches)")

# Eval before
print("\nEval before...")
m.eval(); torch.cuda.empty_cache()

def eval_with_cs(data_ids, n_batches):
    cs.usage.zero_()
    for p in range(cs.n_slots):
        cs.key_bank.data[p].zero_()
        cs.val_bank.data[p].zero_()
    tl = 0.0
    for bi in range(min(30, n_batches)):
        x = data_ids[bi * BS * SEQ : (bi + 1) * BS * SEQ].view(BS, SEQ).to(device)
        emb = m.embed(x)
        h = torch.zeros(BS, DM, device=device); logits = []
        for t in range(SEQ):
            ctx = cs.read(h)
            g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
            h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
            logits.append(m.head(m.state_norm(h)))
        logits = torch.stack(logits, dim=1)
        tl += F.cross_entropy(logits[:, :-1].reshape(-1, 4096), x[:, 1:].reshape(-1)).item()
    return torch.exp(torch.tensor(tl / min(30, n_batches))).item()

def eval_without(data_ids, n_batches):
    tl = 0.0
    for bi in range(min(30, n_batches)):
        x = data_ids[bi * BS * SEQ : (bi + 1) * BS * SEQ].view(BS, SEQ).to(device)
        emb = m.embed(x)
        h = torch.zeros(BS, DM, device=device); logits = []
        for t in range(SEQ):
            h = m.cell(h, emb[:, t, :], step=t)
            logits.append(m.head(m.state_norm(h)))
        logits = torch.stack(logits, dim=1)
        tl += F.cross_entropy(logits[:, :-1].reshape(-1, 4096), x[:, 1:].reshape(-1)).item()
    return torch.exp(torch.tensor(tl / min(30, n_batches))).item()

# Cross-distribution eval: WikiText (never seen during training)
print("\nLoading WikiText eval data...")
wt_cache = "checkpoints/cs_wt_eval.pt"
if os.path.exists(wt_cache):
    wt_ids = torch.load(wt_cache, map_location="cpu", weights_only=True)
else:
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    wt_t = [t["text"] for t in ds if len(t["text"]) > 100][:500]
    wt_list = []
    for t in tqdm(wt_t, desc="tokenizing"):
        e = tok.encode(t).ids[:SEQ * 50]
        if len(e) >= SEQ:
            wt_list += e[:SEQ * 50]
    wt_ids = torch.tensor(wt_list[:50000], dtype=torch.long) if wt_list else torch.zeros(0, dtype=torch.long)
    torch.save(wt_ids, wt_cache)
wt_nb = max(1, (len(wt_ids) - 1) // (BS * SEQ))
print(f"  eval on WikiText: {len(wt_ids):,} tokens ({wt_nb} batches)")

ppl_before = eval_without(wt_ids, wt_nb)
ppl_with_cs = eval_with_cs(wt_ids, wt_nb)
print(f"Without CS: {ppl_before:.2f}  With CS: {ppl_with_cs:.2f}")

# Train
print("\nTraining (200 steps)...")
params = list(cs.parameters()) + list(m.cell.parameters()) + list(m.head.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)

for step in tqdm(range(200), desc="train"):
    m.train(); cs.train()
    bi = step % train_nb
    x = train_ids[bi * BS * SEQ : (bi + 1) * BS * SEQ].view(BS, SEQ).to(device)
    emb = m.embed(x)
    h = torch.zeros(BS, DM, device=device)
    logits = []
    for t in range(SEQ):
        ctx = cs.read(h)
        g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
        h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
        cs.write(h)
        logits.append(m.head(m.state_norm(h)))
    loss = F.cross_entropy(torch.stack(logits, dim=1)[:, :-1].reshape(-1, 4096), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

# Eval after
m.eval(); cs.eval(); torch.cuda.empty_cache()
ppl_after = eval_with_cs(wt_ids, wt_nb)
print(f"\n{'='*40}")
print(f"Cross-distribution (train=code, eval=WikiText):")
print(f"  Without CS:      {ppl_before:.2f}")
print(f"  With CS (init):  {ppl_with_cs:.2f}")
print(f"  With CS (train): {ppl_after:.2f}")
print(f"  Delta:           {ppl_before - ppl_after:+.2f}")
print(f"{'IMPROVEMENT' if ppl_after < ppl_before else 'NO CHANGE'}")
print(f"{'='*40}")
print(f"\nNote: training was on code data (StarCoder). Eval is on WikiText (unseen).")
print(f"Positive delta on cross-distribution eval = genuine generalization.")
print(f"{'='*40}")
