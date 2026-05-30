"""Precise slot: exact token match retrieval vs no slot."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)
SEQ, DM, N_SLOTS = 512, 840, 512

sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, DM, 4096, beta=0.5).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False); m.eval()

class PreciseSlot:
    def __init__(self, d, n):
        self.keys = torch.zeros(n, d, device=device)
        self.vals = torch.zeros(n, d, device=device)
        self.tokens = torch.zeros(n, dtype=torch.long, device=device)
        self.n = n; self.p = 0
    def write(self, h, token_id):
        self.keys[self.p] = h.detach().squeeze()
        self.vals[self.p] = h.detach().squeeze()
        self.tokens[self.p] = token_id
        self.p = (self.p + 1) % self.n
    def read_exact(self, token_id):
        m = (self.tokens == token_id)
        if m.any():
            return self.vals[m.nonzero(as_tuple=True)[0][-1]], True
        return None, False

slot = PreciseSlot(DM, N_SLOTS)
fill = [random.randint(2, 4095) for _ in range(SEQ)]
KEY, VAL = 37, 42
fill[50], fill[51] = KEY, VAL
fill[400], fill[401] = KEY, VAL

x = torch.tensor([fill], dtype=torch.long, device=device)
emb = m.embed(x)

# Pass 1: store
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    slot.write(h, fill[t])
    h = m.cell(h, emb[:, t, :], step=t)

# Pass 2: WITH slot
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    ev, found = slot.read_exact(fill[t])
    if found:
        g = torch.sigmoid((h * ev).sum(dim=-1, keepdim=True))
        h = m.cell(h + g * ev * 0.1, emb[:, t, :], step=t)
    else:
        h = m.cell(h, emb[:, t, :], step=t)
    if t == 400: logit_w = m.head(m.state_norm(h))

# Pass 3: WITHOUT slot
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    h = m.cell(h, emb[:, t, :], step=t)
    if t == 400: logit_wo = m.head(m.state_norm(h))

pw = logit_w[0].argmax().item()
po = logit_wo[0].argmax().item()
print(f"Without slot: pred={po}, target={VAL} {'OK' if po==VAL else 'WRONG'}")
print(f"With slot:    pred={pw}, target={VAL} {'OK' if pw==VAL else 'WRONG'}")
if pw == VAL and po != VAL:
    print("Precise slot ENABLES recall (only works WITH slot)")
elif pw == po == VAL:
    print("Both correct (no slot needed for this sample)")
else:
    print("Both wrong (slot doesn't help for this random sample)")
