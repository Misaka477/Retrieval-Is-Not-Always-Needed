"""Real-text NIAH: needle in WikiText-103 paragraphs."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data crates BEFORE torch (C-extension conflict on Windows)
from tokenizers import Tokenizer
from datasets import load_dataset

import torch, torch.nn.functional as F, time, random
from modules.cann_ssm import RINASeqModel, _full_forward

device = "cuda"; torch.manual_seed(42)

DM, NP, V = 768, 4096, 4096
MAX_SEQ = 512

# ── Load tokenizer + trained model ──
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

def load_model(ckpt_path, ae):
    model = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, n_slots=V, attract_every=ae).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.slot_table.zero_()
    return model

print("Loading models...", flush=True)
cann = load_model("checkpoints/cann_15m_wt103_final.pt", ae=2)
abl  = load_model("checkpoints/cann_15m_abl_final.pt",   ae=9999)
print("  Done.", flush=True)

# ── Load real paragraphs ──
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
paras = []
for t in ds["text"][:10000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 128:
            paras.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(paras)}", flush=True)

# ── Choose RARE BPE tokens as key/value ──
# Very rare tokens (<100) are special chars/subwords that don't appear naturally
KEYS = list(range(1, 6))     # 5 very rare BPE tokens
VALS = list(range(6, 11))    # 5 distinct rare tokens

def make_sample(gap):
    while True:
        p = paras[torch.randint(0, len(paras), (1,)).item()]
        need = 64 + gap + 4
        if len(p) >= need:
            seq = p[:need].tolist()
            k = random.choice(KEYS)
            v = random.choice(VALS)
            seq[0] = k; seq[1] = v; seq[-1] = k
            return torch.tensor(seq), v

def make_batch(n, gap):
    xs, ys = [], []
    for _ in range(n):
        x, v = make_sample(gap)
        xs.append(x)
        ys.append(v)
    return torch.stack(xs), torch.tensor(ys)

def eval_model(model, name, gap, steps=300):
    train_x, train_y = make_batch(200, gap)
    test_x, test_y = make_batch(200, gap)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for step in range(steps):
        model.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 32):
            idx = perm[i:i+32]
            logits = _full_forward(
                train_x[idx].to(device), model.embed.weight, model.slot_table,
                model.head.weight, model.head.bias,
                model.state_norm.weight, model.state_norm.bias,
                model.cell.patterns, model.cell.beta_t,
                model.cell.gate_a.weight, model.cell.gate_a.bias,
                model.cell.gate_b.weight, model.cell.gate_b.bias,
                model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                model.cell.proj_in.weight, model.cell.proj_in.bias,
                model.cell.norm.weight, model.cell.norm.bias,
                model.attract_every,
            )
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()

        with torch.no_grad():
            for b in range(train_x.shape[0]):
                k, v = int(train_x[b, 0]), int(train_y[b])
                if k in KEYS and v in VALS: model.slot_write(k, v)

        if step % 10 == 9:
            model.eval()
            with torch.no_grad():
                lt = _full_forward(
                    test_x.to(device), model.embed.weight, model.slot_table,
                    model.head.weight, model.head.bias,
                    model.state_norm.weight, model.state_norm.bias,
                    model.cell.patterns, model.cell.beta_t,
                    model.cell.gate_a.weight, model.cell.gate_a.bias,
                    model.cell.gate_b.weight, model.cell.gate_b.bias,
                    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                    model.cell.proj_in.weight, model.cell.proj_in.bias,
                    model.cell.norm.weight, model.cell.norm.bias,
                    model.attract_every,
                )
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            prev = best
            best = max(best, acc)
            print(f"  {name} gap={gap:3d} step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")
            if best >= 1.0: break
            if step > 100 and best == prev:  # plateau, skip
                print(f"  {name} gap={gap:3d} PLATEAU at step={step+1}, best={best*100:.0f}%")
                break
            model.train()
    return best

print("\n  gap    CANN+slot    ABL+slot    delta")
print("  ─────────────────────────────────────")
for gap in [8, 16, 32, 64, 128]:
    cann.slot_table.zero_(); abl.slot_table.zero_()
    bc = eval_model(cann, "CANN", gap)
    ba = eval_model(abl,  "ABL ", gap)
    print(f"  {gap:3d}     {bc*100:.0f}%         {ba*100:.0f}%         {bc-ba:+.0f}%")
print()
