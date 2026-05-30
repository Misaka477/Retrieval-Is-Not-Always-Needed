"""
Phase 3a: Train RINA LM on real text (Pride and Prejudice).

Character-level vocabulary, next-token prediction.
Compares Hopfield LM vs CANN-SSM + Slot.
"""
import torch, torch.nn.functional as F, sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "references", "hopfield-layers"))
from hflayers import Hopfield

device = "cuda"

# ── Tokenizer ──
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\";:()\n"
stoi = {c: i+3 for i, c in enumerate(CHARS)}
stoi['<pad>'] = 0; stoi['<bos>'] = 1; stoi['<eos>'] = 2
itos = {i: c for c, i in stoi.items()}
V = len(stoi)  # ~75
print(f"Vocab size: {V}")

def encode(text): return torch.tensor([stoi.get(c, stoi['<pad>']) for c in text])
def decode(ids): return "".join(itos.get(i, '') for i in ids if i > 2)

# ── Load P&P ──
pp_path = "D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt"
pp_text = open(pp_path, "r", encoding="latin-1").read()
chap_idx = pp_text.find("CHAPTER")
if chap_idx >= 0:
    pp_text = pp_text[chap_idx:chap_idx+50000]

data = encode(pp_text)
print(f"P&P chars: {len(data):,}")

# ── Models ──
class HopfieldLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=4096):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.hopfield = Hopfield(input_size=d_model, hidden_size=d_model, output_size=d_model,
                                 num_heads=1, scaling=0.5, update_steps_max=3, batch_first=True)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        return self.head(self.norm(self.hopfield(self.embed(x))))

from modules.cann_ssm import RINASeqModel as CANNSSM_Model

# ── Training ──
def make_batches(data, seq_len=128, batch_size=32):
    n = len(data) - seq_len - 1
    idxs = torch.randperm(n)[:batch_size * (n // batch_size)].reshape(-1, batch_size)
    for idx_batch in idxs:
        x = torch.stack([data[i:i+seq_len] for i in idx_batch])
        y = torch.stack([data[i+1:i+seq_len+1] for i in idx_batch])
        yield x.to(device), y.to(device)

d_model = 128
n_patterns = 2048

name = "Hopfield"
t0 = time.time()
model = HopfieldLM(V, d_model=d_model, n_patterns=n_patterns).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n{name}: {n_params:,} params")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_loss = float('inf')

for ep in range(80):
    model.train()
    total_loss = 0; n_batches = 0
    for x, y in make_batches(data, seq_len=128, batch_size=32):
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    best_loss = min(best_loss, avg_loss)
    if ep % 10 == 9:
        print(f"  ep={ep:2d}: loss={avg_loss:.4f} ppl={ppl:.2f} ({time.time()-t0:.0f}s)")

print(f"\n{name}: best_loss={best_loss:.4f} ({(time.time()-t0)/60:.1f}min)")

# Generate sample
model.eval()
prompt = "I just told you a secret password. The password is "
x = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
print(f"\nPrompt: {prompt}")
with torch.no_grad():
    for _ in range(30):
        logits = model(x)[0, -1]
        nid = logits.argmax().item()
        c = itos.get(nid, '?')
        x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
        print(c, end='', flush=True)
print()
