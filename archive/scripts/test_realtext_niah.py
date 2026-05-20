"""
Real-text NIAH: P&P haystack + KILO42 needle.

Character-level tokenization (our 23-vocab toy model).
Needle = "KILO42" embedded in real text.
Query = "password" cue.
"""
import torch, torch.nn.functional as F, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "references", "hopfield-layers"))
from hflayers import Hopfield

device = "cuda"


class HopfieldLM(torch.nn.Module):
    """Simple next-character prediction with Hopfield."""
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.hopfield = Hopfield(input_size=d_model, hidden_size=d_model, output_size=d_model,
                                 num_heads=1, scaling=0.5, update_steps_max=3, batch_first=True)
        self.head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.hopfield(self.embed(x)))


# ── Build character vocab ──
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\";:()\n"
stoi = {c: i+3 for i, c in enumerate(CHARS)}
stoi['<pad>'] = 0; stoi['<bos>'] = 1; stoi['<eos>'] = 2
itos = {i: c for c, i in stoi.items()}
vocab_size = len(stoi)
print(f"Vocab size: {vocab_size}")

def encode(text): return [stoi.get(c, stoi['<pad>']) for c in text]
def decode(ids): return "".join(itos.get(i, '') for i in ids)

# ── Load P&P ──
pp_path = "D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt"
pp_text = open(pp_path, "r", encoding="latin-1").read()
chap_idx = pp_text.find("CHAPTER")
if chap_idx >= 0:
    pp_text = pp_text[chap_idx:chap_idx+100000]
pp_ids = encode(pp_text)
print(f"P&P chars available: {len(pp_ids):,}")

# ── Needle setup ──
needle_str = "KILO42"
query_str = "password"
nd_ids = encode(needle_str)
q_ids = encode(query_str)
print(f"Needle: {nd_ids} ({needle_str})")
print(f"Query: {q_ids} ({query_str})")

# ── Train model ──
def make_train_data(pp_ids, n_seqs=500, seq_len=128):
    """Create NIAH training sequences from P&P.

    Sequence: [P&P text with needle] + [query] + [needle_char_1, needle_char_2, ...]
    The model must predict the next character after the query.
    """
    xs, ys = [], []
    for _ in range(n_seqs):
        start = torch.randint(0, max(1, len(pp_ids) - seq_len - len(nd_ids) - len(q_ids)), (1,)).item()
        seg = pp_ids[start:start + seq_len]
        nd_pos = torch.randint(0, max(1, seq_len - len(nd_ids)), (1,)).item()
        seg[nd_pos:nd_pos + len(nd_ids)] = nd_ids
        full_seq = seg + q_ids + nd_ids
        # Input: full_seq[:-1] (all but last char)
        # Target: full_seq[1:]  (shifted right by 1)
        inp = full_seq[:-1]
        tgt = full_seq[1:]
        xs.append(inp)
        ys.append(tgt)
    return torch.tensor(xs), torch.tensor(ys)


print("\nTraining model on P&P NIAH tasks...")
xi, yi = make_train_data(pp_ids, n_seqs=500, seq_len=128)
seq_len = xi.shape[1]

model = HopfieldLM(vocab_size, d_model=64).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for ep in range(80):
    model.train()
    for i in range(0, 500, 16):
        x = xi[i:i+16].to(device)
        y = yi[i:i+16].to(device)
        loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(xi[:100].to(device))[:, -1].argmax(dim=-1) == yi[:100, -1].to(device)).float().mean().item()
    if ep % 20 == 19:
        print(f"  ep={ep}: last_char_acc={acc*100:.0f}%")

# ── Real-text NIAH evaluation ──
print("\n=== Real-text NIAH Evaluation ===")

context_lens = [256, 512, 1024]
results = {}

for ctx_len in context_lens:
    for depth in [0.25, 0.5, 0.75]:
        needle_tok_pos = int(ctx_len * depth)
        needle_tok_pos = max(0, min(needle_tok_pos, len(pp_ids) - ctx_len - 100))

        base = pp_ids[needle_tok_pos:needle_tok_pos + ctx_len].copy()
        nd_start = max(0, ctx_len // 2)
        nd_end = min(nd_start + len(nd_ids), ctx_len)
        base[nd_start:nd_end] = nd_ids[:nd_end - nd_start]
        base.extend(q_ids)

        if len(base) > ctx_len + len(q_ids):
            base = base[:ctx_len + len(q_ids)]

        x = torch.tensor([base], device=device)
        gen_ids = []
        with torch.no_grad():
            for step in range(20):
                logits = model(x)
                next_id = logits[0, -1].argmax().item()
                gen_ids.append(next_id)
                if next_id in (0, 1, 2):
                    break
                x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

        gen_text = decode(gen_ids)
        passed = needle_str.upper() in gen_text.upper()
        results[f"ctx{ctx_len}_d{depth}"] = {
            "pass": passed, "pred": gen_text,
            "needle_pos": nd_start, "ctx_len": ctx_len, "depth": depth
        }

        mark = "PASS" if passed else "FAIL"
        print(f"  ctx={ctx_len:4d} depth={depth:.2f}: {mark} gen={repr(gen_text)}")

pass_count = sum(1 for v in results.values() if v["pass"])
total = len(results)
print(f"\nResults: {pass_count}/{total} PASS")
print(json.dumps(results, indent=2))
