"""
位置感知 Context Slot + 真实文本（WikiText）。
测 slot 是否能在真实语言中改变预测分布。
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)
SEQ, DM, N_SLOTS = 512, 840, 256

sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, DM, 4096, beta=0.5).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False); m.eval()

class PosSlot:
    def __init__(self, d, n):
        self.keys = torch.zeros(n, d, device=device)
        self.vals = torch.zeros(n, d, device=device)
        self.n = n; self.p = 0
    def write(self, h):
        self.keys[self.p] = h.detach().squeeze()
        self.vals[self.p] = h.detach().squeeze()
        self.p = (self.p + 1) % self.n
    def read(self, h, step, beta=2.0, gamma=0.1):
        scores = h @ self.keys.T * beta
        pos_bias = -gamma * torch.abs(torch.arange(self.n, device=device) - step).float()
        a = F.softmax(scores + pos_bias, dim=-1)
        return a @ self.vals

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
text = [t["text"] for t in ds if len(t["text"]) > 200][0]
ids = tok.encode(text).ids[:min(len(tok.encode(text).ids), 512)]
SEQ = len(ids)
x = torch.tensor([ids], dtype=torch.long, device=device)
emb = m.embed(x)
print(f"Real text: {text[:120]}...")
print(f"Tokens: {SEQ}")

# Store h_t
cs = PosSlot(DM, N_SLOTS)
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    cs.write(h)
    h = m.cell(h, emb[:, t, :], step=t)

# Mid-sequence query
q_pos = SEQ // 2
h = torch.zeros(1, DM, device=device)
logit_w = None
for t in range(SEQ):
    ctx = cs.read(h, t, gamma=0.1)
    g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
    h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
    if t == q_pos: logit_w = m.head(m.state_norm(h))

h = torch.zeros(1, DM, device=device)
logit_wo = None
for t in range(SEQ):
    h = m.cell(h, emb[:, t, :], step=t)
    if t == q_pos: logit_wo = m.head(m.state_norm(h))

tw = logit_w[0].topk(3).indices.tolist()
wo = logit_wo[0].topk(3).indices.tolist()
diff = (logit_w - logit_wo).abs().mean().item()
overlap = len(set(tw) & set(wo))
target = x[0, q_pos + 1].item()  # 模型在 q_pos 处预测的目标 token
correct_w = (logit_w[0].argmax().item() == target)
correct_wo = (logit_wo[0].argmax().item() == target)

print(f"\nPos {q_pos}: real text eval")
print(f"  Top-3 WO: {wo}")
print(f"  Top-3 W:  {tw}")
print(f"  Overlap:  {overlap}/3  Diff: {diff:.3f}")
print(f"  Correct (target={target}) — WO: {correct_wo}, W: {correct_w}")
print(f"  {'CHANGED' if overlap < 3 else 'UNCHANGED'}")
