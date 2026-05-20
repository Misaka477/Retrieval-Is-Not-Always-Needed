"""Train Hopfield LM on Pride and Prejudice, generate sample."""
import torch, torch.nn.functional as F, sys, math, time
sys.path.insert(0, "D:\\Software_Development\\Project\\RINA_Project\\references\\hopfield-layers")
from hflayers import Hopfield
device = "cuda"

# Build character vocab
import string
chars = string.printable.strip()
stoi = {c: i+3 for i, c in enumerate(chars)}
stoi["<pad>"] = 0; stoi["<bos>"] = 1; stoi["<eos>"] = 2
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

# Load data
pp = open("D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt",
          "r", encoding="latin-1").read()
ci = pp.find("CHAPTER")
pp = pp[ci:ci+20000] if ci >= 0 else pp[:20000]
data = torch.tensor([stoi.get(c, 0) for c in pp])
print(f"Data: {len(data):,} chars, vocab={V}")

# Model
dm = 64
m = torch.nn.Sequential(
    torch.nn.Embedding(V, dm),
    Hopfield(dm, dm, dm, num_heads=1, scaling=0.5, update_steps_max=3, batch_first=True),
    torch.nn.LayerNorm(dm),
    torch.nn.Linear(dm, V),
).to(device)
print(f"Params: {sum(p.numel() for p in m.parameters()):,}")

opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
t0 = time.time()
for ep in range(60):
    m.train()
    tl = 0
    for _ in range(100):
        idx = torch.randint(0, len(data) - 65, (32,))
        x = torch.stack([data[j:j+64] for j in idx]).to(device)
        y = torch.stack([data[j+1:j+65] for j in idx]).to(device)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()
    ppl = math.exp(tl / 100)
    if ep % 10 == 9:
        print(f"ep={ep}: loss={tl/100:.4f} ppl={ppl:.2f} ({time.time()-t0:.0f}s)")

# Generate
m.eval()
prompt = "password "
x = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
with torch.no_grad():
    for _ in range(100):
        nid = m(x)[0, -1].argmax().item()
        ch = itos.get(nid, "?")
        print(ch, end="", flush=True)
        x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
print()
