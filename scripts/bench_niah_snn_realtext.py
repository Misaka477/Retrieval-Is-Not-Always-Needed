"""Real-text NIAH on SNN v2 — 对标 V1 bench_niah_realtext."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset

import torch, torch.nn.functional as F, time, random
from modules.temporal_snn_cell import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)

DM, NP, V = 840, 4096, 4096

# ── Load tokenizer ──
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
V_tok = tok.get_vocab_size()
print(f"Tokenizer: vocab={V_tok}", flush=True)

# ── Load SNN model ──
CKPT = "checkpoints/code_seq256_resume.pt"
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = TemporalSNNModel(V_tok, d_model=DM, n_patterns=NP, beta=0.5, attract_every=2,
                          error_threshold=1.0, hebbian_lr=0.0,
                          inhibition_threshold=0.0).to(device)
model.load_state_dict(ckpt["model"], strict=False)
model.train()

# Trainable slot_proj
model.slot_proj = torch.nn.Linear(DM, DM).to(device)
model.slot_proj.weight.data.normal_(0, 0.01)
model.slot_proj.bias.data.zero_()

# ── Load real paragraphs ──
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
paras = []
for t in ds["text"][:10000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 128:
            paras.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(paras)}", flush=True)

# ── Rare BPE tokens as key/value (same as V1) ──
KEYS = list(range(1, 6))
VALS = list(range(6, 11))

def make_sample(gap):
    while True:
        p = paras[torch.randint(0, len(paras), (1,)).item()]
        need = 64 + gap + 4
        if len(p) >= need:
            seq = p[:need].tolist()
            k = random.choice(KEYS); v = random.choice(VALS)
            seq[0] = k; seq[1] = v; seq[-1] = k
            return torch.tensor(seq), v

def make_batch(n, gap):
    xs, ys = [], []
    for _ in range(n):
        x, v = make_sample(gap)
        xs.append(x); ys.append(v)
    return torch.stack(xs), torch.tensor(ys)

def forward_with_slot(model, x, slot_dict):
    bsz, sl = x.shape; dm = model.d_model
    emb = model.embed(x); h = torch.zeros(bsz, dm, device=device)
    logits = []
    for t in range(sl):
        if t < sl - 1:
            h = model.cell(h, emb[:, t, :], step=t)
        else:
            inj = torch.stack([slot_dict.get(x[b, -1].item(), torch.zeros(dm, device=device)) for b in range(bsz)])
            h = model.cell(h + inj, emb[:, t, :], step=t)
            pat = model.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h.unsqueeze(1)
            sc = xi @ pat.transpose(1, 2) * model.cell.beta_t[0]
            attracted = (torch.softmax(sc, dim=-1) @ pat).squeeze(1)
            cl = torch.cat([h, emb[:, -1, :]], dim=-1)
            alpha = torch.sigmoid(model.cell.gate_alpha(cl))
            h = h + alpha * (attracted - h); h = model.cell.norm(h)
        logits.append(model.head(model.state_norm(h)))
    return torch.stack(logits, dim=1)

def eval_model(model, name, gap, steps=300):
    train_x, train_y = make_batch(200, gap)
    test_x, test_y = make_batch(200, gap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for step in range(steps):
        model.train()
        opt.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 32):
            idx = perm[i:i+32]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)

            slot = {}
            for b in range(len(idx)):
                k, v = int(xb[b, 0]), int(yb[b])
                if k in KEYS and v in VALS:
                    slot[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)

            logits = forward_with_slot(model, xb, slot)
            loss = F.cross_entropy(logits[:, -1], yb)
            loss.backward()
        opt.step()

        if step % 10 == 9:
            model.eval()
            with torch.no_grad():
                slot_test = {}
                for b in range(test_x.shape[0]):
                    k, v = int(test_x[b, 0]), int(test_y[b])
                    if k in KEYS and v in VALS:
                        slot_test[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)
                lt = forward_with_slot(model, test_x.to(device), slot_test)
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            prev = best
            best = max(best, acc)
            print(f"  {name} gap={gap:3d} step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")
            if best >= 1.0: break
            if step > 100 and best == prev:
                print(f"  {name} gap={gap:3d} PLATEAU at step={step+1}, best={best*100:.0f}%")
                break
            model.train()
    return best

print("\n  gap    SNN+slot")
print("  ────────────────")
for gap in [8, 16, 32, 64, 128]:
    best = eval_model(model, "SNN", gap)
    print(f"  {gap:3d}     {best*100:.0f}%")
print()
