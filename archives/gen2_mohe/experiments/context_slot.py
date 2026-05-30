"""
Context Slot 极简验证：存 h_t（情境向量）而非 token→token.
所有 h_t 逐一存储，每步检索最相关过去状态。
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)
SEQ = 1024; N_SLOTS = 64

sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5,
                      attract_every=2, error_threshold=1.0, hebbian_lr=0.0).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False); m.eval()
print(f"Model: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

class ContextSlot:
    def __init__(self, d, n):
        self.keys = torch.zeros(n, d, device=device)
        self.vals = torch.zeros(n, d, device=device)
        self.n = n; self.p = 0
    def write(self, h):
        self.keys[self.p] = h.detach().squeeze()
        self.vals[self.p] = h.detach().squeeze()
        self.p = (self.p + 1) % self.n
    def read(self, h, beta=2.0):
        a = F.softmax(h @ self.keys.T * beta, dim=-1)
        return a @ self.vals

slot = ContextSlot(840, N_SLOTS)
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
text = next(iter(code))["content"]
ids = tok.encode(text).ids
ids = ids[:min(len(ids), 1024)]
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
SEQ = len(ids)
print(f"Tokens: {SEQ}")

emb = m.embed(x)
# Store all h_t in context slot (sequential pass)
h = torch.zeros(1, 840, device=device)
for t in range(SEQ):
    slot.write(h)
    ctx = slot.read(h)
    g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
    h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)

# Evaluate WITH context
h = torch.zeros(1, 840, device=device); tl_w = 0.0
for t in range(SEQ):
    ctx = slot.read(h)
    g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
    h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
    tl_w += F.cross_entropy(m.head(m.state_norm(h)), x[:, t].view(-1), reduction='sum').item()

# Evaluate WITHOUT context
h = torch.zeros(1, 840, device=device); tl_wo = 0.0
for t in range(SEQ):
    h = m.cell(h, emb[:, t, :], step=t)
    tl_wo += F.cross_entropy(m.head(m.state_norm(h)), x[:, t].view(-1), reduction='sum').item()

ppl_w = torch.exp(torch.tensor(tl_w / SEQ)).item()
ppl_wo = torch.exp(torch.tensor(tl_wo / SEQ)).item()
d = ppl_wo - ppl_w
print(f"Without: {ppl_wo:.2f}  With: {ppl_w:.2f}  Delta: {d:+.2f}  {'HELPS' if d > 0 else 'HURTS'}")
