"""
K pre-rotate gate input 测试.

当前:  h_ssm = gate(h, x) → attractor(h_ssm) → h_new
方案:  h_pre = K@h   → h_new_k = gate(h_pre, x)  (无 attractor)

核心问题: K@h 能否取代 attractor(h) 作为 gate 的输入？
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 加载真实模型 ──
from modules.cann_ssm import RINASeqModel
import modules.cann_ssm as _c; _c._setup_cuda_seq_v2 = lambda: False

dm, V = 768, 4096
CKPT = "checkpoints/cann_lowrank_ep10.pt"
if not os.path.exists(os.path.join(_ROOT, CKPT)):
    CKPT = "checkpoints/cann_lowrank_final.pt"
print(f"Loading: {CKPT}")
st = torch.load(os.path.join(_ROOT, CKPT), map_location=device, weights_only=True)
model = RINASeqModel(V, d_model=dm, n_patterns=4096, beta=0.5, n_slots=V,
                     attract_every=2, pattern_rank=128).to(device)
model.load_state_dict(st, strict=False)
model.eval()

# ── 提取 patterns ──
if model.cell.patterns is not None:
    patterns = model.cell.patterns.detach()
else:
    patterns = (model.cell.U @ model.cell.V).detach()
print(f"  patterns: {patterns.shape}")

# ── 提取 gate 权重 ──
Wa = model.cell.gate_a.weight.detach()
ba = model.cell.gate_a.bias.detach()
Wb = model.cell.gate_b.weight.detach()
bb = model.cell.gate_b.bias.detach()
Wp = model.cell.proj_in.weight.detach()
bp = model.cell.proj_in.bias.detach()
Wn = model.cell.norm.weight.detach()
bn = model.cell.norm.bias.detach()

# ── 拟合 K (DMD) ──
print("Fitting K...")
N = 8192; torch.manual_seed(42)
idx = torch.randint(0, patterns.shape[0], (N,), device=device)
h_src = patterns[idx] + torch.randn(N, dm, device=device) * 0.1
h_src = h_src / h_src.norm(dim=-1, keepdim=True)
scores = (h_src @ patterns.T) * 0.5
attn = torch.softmax(scores, dim=-1)
alpha_val = 0.1
y_att = h_src + alpha_val * (attn @ patterns - h_src)
K = torch.linalg.lstsq(h_src, y_att, rcond=None).solution.T
print(f"  K shape: {K.shape}")


# ── 两种 gate 实现 ──
def gate(h, x):
    """原始 gate: h_ssm = a*h + b*xp, LayerNorm 输出"""
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ Wa.T + ba)
    b = torch.sigmoid(combined @ Wb.T + bb)
    xp = x @ Wp.T + bp
    h_ssm = a * h + b * xp
    return F.layer_norm(h_ssm, [dm], Wn, bn, eps=1e-5)


def attractor(h, patterns):
    """非线性 attractor (用于参考)"""
    scores = (h @ patterns.T) * 0.5
    attn = torch.softmax(scores, dim=-1)
    attracted = attn @ patterns
    return h + 0.1 * (attracted - h)


def gate_with_K_pre(h, x, K_mat):
    """K 预旋转: gate(K@h, x)"""
    h_pre = h @ K_mat.T
    return gate(h_pre, x)


# ── 序列对比测试 ──
print("\n── Sequence rollout: original vs K pre-rotate ──")
seq_len = 128
bs = 8
torch.manual_seed(42)
tokens = torch.randint(0, V, (bs, seq_len), device=device)
emb = model.embed(tokens)

# Path A: 原始 gate→attractor
h_a = torch.zeros(bs, dm, device=device)
h_a_traj = []

# Path B: K@h → gate
h_b = torch.zeros(bs, dm, device=device)
h_b_traj = []

cos_ab = []       # cos(gate_a_out, gate_b_out)
cos_a_in = []     # cos(h_a, K@h_a)  K 的预旋转偏差

for t in range(seq_len):
    x = emb[:, t, :]

    # Path A: gate(h, x) → attractor
    h_a_out = gate(h_a, x)
    h_a_att = attractor(h_a_out, patterns)
    h_a = F.layer_norm(h_a_att, [dm], Wn, bn, eps=1e-5)
    h_a_traj.append(h_a.clone())

    # Path B: K@h → gate
    h_b_pre = h_b @ K.T
    h_b_out = gate_with_K_pre(h_b, x, K)
    h_b = h_b_out
    h_b_traj.append(h_b.clone())

    if h_a_out.norm() > 1e-8 and h_b_out.norm() > 1e-8:
        cos_ab.append(F.cosine_similarity(h_a_out, h_b_out, dim=-1).mean().item())
    cos_a_in.append(F.cosine_similarity(h_a @ K.T, h_a_att, dim=-1).mean().item())

print(f"  Steps: {len(cos_ab)}")
print(f"  cos(gate_A_out, gate_B_out):")
print(f"    mean={statistics.mean(cos_ab):.4f}  median={statistics.median(cos_ab):.4f}")
print(f"    early(0-15)={statistics.mean(cos_ab[:16]):.4f}  mid(48-63)={statistics.mean(cos_ab[48:64]):.4f}  late(112-127)={statistics.mean(cos_ab[-16:]):.4f}")
print(f"  cos(K@h, attractor(h)):  mean={statistics.mean(cos_a_in):.4f}")

# ── 轨迹累散 ──
h_a_stack = torch.stack(h_a_traj)
h_b_stack = torch.stack(h_b_traj)
cross_cos = F.cosine_similarity(
    h_a_stack.reshape(-1, dm), h_b_stack.reshape(-1, dm), dim=-1
).reshape(seq_len, bs).mean(dim=-1)

print(f"\n── Full trajectory cos(h_A_t, h_B_t) ──")
for s in [0, 3, 7, 15, 31, 63, 127]:
    print(f"  step {s+1:4d}: {cross_cos[min(s, seq_len-1)]:.4f}")

final_cos = cross_cos[-1].item()
if final_cos > 0.95:
    verdict = "PASS — K pre-rotate preserves gate output"
elif final_cos > 0.85:
    verdict = "MARGINAL — significant drift over 128 steps"
else:
    verdict = f"FAIL — states diverged (final cos={final_cos:.3f})"
print(f"\n  Verdict: {verdict}")


# ── 消融: K 替换程度 ──
print(f"\n── Ablation: hybrid K fraction ──")
# 每 M 步用真实 attractor 同步一次
for sync_every in [1, 2, 4, 8, 16, 32, 64, 128]:
    h_hyb = torch.zeros(bs, dm, device=device)
    h_ref = torch.zeros(bs, dm, device=device)
    hyb_cos = []
    for t in range(seq_len):
        x = emb[:, t, :]
        h_ref_out = gate(h_ref, x)
        h_ref = F.layer_norm(attractor(h_ref_out, patterns), [dm], Wn, bn, eps=1e-5)

        if t % sync_every == 0:
            h_hyb = h_ref.clone()  # sync
        h_hyb_out = gate_with_K_pre(h_hyb, x, K)
        h_hyb = h_hyb_out
        hyb_cos.append(F.cosine_similarity(h_hyb, h_ref, dim=-1).mean().item())

    avg_end = statistics.mean(hyb_cos[-16:])
    print(f"  sync_every={sync_every:3d}:  late cos={avg_end:.4f}  "
          f"({'stable' if avg_end > 0.9 else 'drifting' if avg_end > 0.7 else 'collapsed'})")
