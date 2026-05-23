"""
位置感知 Context Slot 测试：softmax 权重加入位置邻近度偏置。
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)
SEQ, DM, N_SLOTS = 512, 840, 256

sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, DM, 4096, beta=0.5).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False); m.eval()

class PositionAwareSlot:
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
        attn = F.softmax(scores + pos_bias, dim=-1)
        return attn @ self.vals

cs = PositionAwareSlot(DM, N_SLOTS)
fill = [random.randint(2, 4095) for _ in range(SEQ)]
x = torch.tensor([fill], dtype=torch.long, device=device)
emb = m.embed(x)

# 存 h_t
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    cs.write(h)
    h = m.cell(h, emb[:, t, :], step=t)

# 测位置感知检索
h = torch.zeros(1, DM, device=device)
logit_with = None
for t in range(SEQ):
    ctx = cs.read(h, step=t, gamma=0.1)
    g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
    h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
    if t == 450:
        logit_with = m.head(m.state_norm(h))

# 无 slot 基线
h = torch.zeros(1, DM, device=device)
logit_wo = None
for t in range(SEQ):
    h = m.cell(h, emb[:, t, :], step=t)
    if t == 450:
        logit_wo = m.head(m.state_norm(h))

tw = logit_with[0].topk(5).indices.tolist()
wo = logit_wo[0].topk(5).indices.tolist()
diff = (logit_with - logit_wo).abs().mean().item()
overlap = len(set(tw) & set(wo))

print(f"Position-aware slot at pos 450:")
print(f"  Top-5 WO: {wo}")
print(f"  Top-5 W:  {tw}")
print(f"  Overlap:  {overlap}/5")
print(f"  Logit diff: {diff:.4f}")
print(f"  {'CHANGED' if overlap < 5 else 'UNCHANGED'}")
print(f"\nNote: gamma=0.1, position bias adds -gamma*|slot_pos - cur_step| to softmax scores.")
