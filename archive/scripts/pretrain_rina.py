"""
RINA Phase 3a: Pre-train on real text.

Trains a Hopfield LM + GPU slot table on Pride and Prejudice.
BPE tokenizer (vocab=4096), next-token prediction.
Runs unattended (~hours).
"""
import torch, torch.nn.functional as F, sys, os, math, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "D:\\Software_Development\\Project\\RINA_Project\\references\\hopfield-layers")
from hflayers import Hopfield
from tokenizers import ByteLevelBPETokenizer

device = "cuda"
CKPT = "D:\\Software_Development\\Project\\RINA_Project\\checkpoints"
os.makedirs(CKPT, exist_ok=True)
torch.manual_seed(42)

# ── Config ──
V = 4096
DM = 256
NP = 4096
SEQ = 128
BS = 16
LR = 3e-4
EPOCHS = 60
BETA = 0.5

print(f"vocab={V} d_model={DM} n_patterns={NP} seq={SEQ} batch={BS}")

# ── Step 1: Tokenizer ──
pp = "D:\\Software_Development\\Project\\RINA_Core\\scripts\\evaluation\\_pride.txt"
tok = ByteLevelBPETokenizer()
tok.train([pp], vocab_size=V, min_frequency=2,
          special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
tok.save_model(CKPT, "rina_4096")
V_actual = tok.get_vocab_size()
print(f"Tokenizer vocab: {V_actual}")

# ── Step 2: Tokenize text ──
text = open(pp, "r", encoding="latin-1").read()
ci = text.find("CHAPTER")
if ci >= 0:
    text = text[ci:]
data = torch.tensor(tok.encode(text).ids, dtype=torch.long, device="cpu")
print(f"Tokens: {len(data):,}")

# ── Step 3: Model ──
class HopfieldLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V_actual, DM)
        self.hopfield = Hopfield(DM, DM, DM, num_heads=1,
                                 scaling=BETA, update_steps_max=3,
                                 batch_first=True)
        self.norm = torch.nn.LayerNorm(DM)
        self.head = torch.nn.Linear(DM, V_actual)
        self.register_buffer("slot_tab", torch.zeros(V_actual, DM))
        self.slot_proj = torch.nn.Linear(DM, DM)

    def forward(self, x):
        return self.head(self.norm(self.hopfield(self.embed(x))))

model = HopfieldLM().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 4: Train ──
opt = torch.optim.AdamW(model.parameters(), lr=LR)
batch_per_epoch = (len(data) - SEQ - 1) // BS
t0 = time.time()

for ep in range(EPOCHS):
    model.train()
    total_loss = 0
    perm = torch.randperm(len(data) - SEQ - 1, device="cpu")

    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        x = torch.stack([data[j:j+SEQ] for j in idx]).to(device)
        y = torch.stack([data[j+1:j+SEQ+1] for j in idx]).to(device)

        loss = F.cross_entropy(model(x).reshape(-1, V_actual), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    ppl = math.exp(total_loss / batch_per_epoch)

    if ep % 5 == 4 or ep == EPOCHS - 1:
        model.eval()
        prompt = "The secret password is"
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            for _ in range(40):
                logits = model(x)
                nid = logits[0, -1].argmax().item()
                x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
        gen = tok.decode(x[0].tolist())
        wall = (time.time() - t0) / 60
        print(f"ep={ep:3d}  ppl={ppl:.2f}  loss={total_loss/batch_per_epoch:.4f}  ({wall:.1f}min)")
        print(f"  gen: {gen[:120]}")

torch.save(model.state_dict(), f"{CKPT}/rina_lm_4096.pt")
print(f"\nDone ({time.time()-t0:.0f}s). Saved to {CKPT}/rina_lm_4096.pt")
