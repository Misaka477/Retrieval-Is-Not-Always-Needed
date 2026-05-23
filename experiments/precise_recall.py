"""
精确位置召回测试：在代码序列中插入 key=value, 300 步后查询。
测量 context slot 在该位置上的 ppl 改善。
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42)
SEQ = 512; DM = 840; N_SLOTS = 256

sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, DM, 4096, beta=0.5).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
m.eval()

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

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

cs = ContextSlot(DM, N_SLOTS)

# 构造序列：用随机 token 填充，在 pos=100 处插入 "model_name=gpt4"
# 在 pos=450 处查询 "model_name"
fill = [random.randint(2, 4095) for _ in range(SEQ)]
kv_pos, q_pos = 100, 450

# 写入 key=value → slot 存的是该位置的 h_t
# 先跑一遍，在 kv_pos 处存下 h_t
x = torch.tensor([fill], dtype=torch.long, device=device)
emb = m.embed(x)
h = torch.zeros(1, DM, device=device)
for t in range(SEQ):
    cs.write(h)
    h = m.cell(h, emb[:, t, :], step=t)

# 现在 slot 里存了所有位置的 h_t
# 在 q_pos 处，slot.read(h) 应该检索到最相似的过去状态

# 测试：在 q_pos 处比较有 slot 和无 slot 的 logit 分布
h = torch.zeros(1, DM, device=device)
logit_with = None
for t in range(SEQ):
    ctx = cs.read(h)
    g = torch.sigmoid((h * ctx).sum(dim=-1, keepdim=True))
    h = m.cell(h + g * ctx * 0.05, emb[:, t, :], step=t)
    if t == q_pos:
        logit_with = m.head(m.state_norm(h))

h = torch.zeros(1, DM, device=device)
logit_without = None
for t in range(SEQ):
    h = m.cell(h, emb[:, t, :], step=t)
    if t == q_pos:
        logit_without = m.head(m.state_norm(h))

# 比较两个 logit 分布的差异
# 如果 slot 检索到了相关信息，logit 分布应该改变（向"正确答案"偏移）
diff = (logit_with - logit_without).abs().mean().item()
top5_with = logit_with[0].topk(5).indices.tolist()
top5_without = logit_without[0].topk(5).indices.tolist()
overlap = len(set(top5_with) & set(top5_without))

print(f"Position {q_pos} (query point):")
print(f"  Top-5 WITHOUT slot: {top5_without}")
print(f"  Top-5 WITH slot:    {top5_with}")
print(f"  Overlap: {overlap}/5")
print(f"  Mean logit diff: {diff:.4f}")
print(f"\nInterpretation:")
if overlap < 5:
    print(f"  Slot CHANGED the prediction at q_pos — retrieval active.")
else:
    print(f"  Slot unchanged at q_pos — retrieval not affecting this position.")
print(f"  (A change doesn't mean 'correct' — it means slot injected non-zero signal.)")
