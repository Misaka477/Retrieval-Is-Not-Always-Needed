"""
双流解耦验证 — attractor 拉回后 K 能维持多少步？

核心问题:
  1. Full attractor 把 h 拉到 cos=0.999（on manifold）
  2. 后续 N 步用 K 维持（gate(K@h, x) → 新 h）
  3. cos_sim(h, nearest_basin) 下降到 0.9 需要多少步？
  4. 如果能维持 ≥5 步 → attractor 调用频率降 5×
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

patterns = (model.cell.U @ model.cell.V).detach() if model.cell.patterns is None else model.cell.patterns.detach()
Wa = model.cell.gate_a.weight.detach(); ba = model.cell.gate_a.bias.detach()
Wb = model.cell.gate_b.weight.detach(); bb = model.cell.gate_b.bias.detach()
Wp = model.cell.proj_in.weight.detach(); bp = model.cell.proj_in.bias.detach()
Wn = model.cell.norm.weight.detach(); bn = model.cell.norm.bias.detach()
print(f"  patterns: {patterns.shape}")


def gate_func(h, x):
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ Wa.T + ba)
    b = torch.sigmoid(combined @ Wb.T + bb)
    xp = x @ Wp.T + bp
    h_ssm = a * h + b * xp
    return F.layer_norm(h_ssm, [dm], Wn, bn, eps=1e-5)

def attractor_func(h):
    scores = (h @ patterns.T) * 0.5
    attn = torch.softmax(scores, dim=-1)
    attracted = attn @ patterns
    return h + 0.1 * (attracted - h)

def basin_cos(h):
    h_n = F.normalize(h, dim=-1)
    p_n = F.normalize(patterns, dim=-1)
    return (h_n @ p_n.T).max(dim=-1).values.mean().item()


print("Fitting K (DMD)...")
N = 8192; torch.manual_seed(42)
idx = torch.randint(0, patterns.shape[0], (N,), device=device)
h_src = patterns[idx] + torch.randn(N, dm, device=device) * 0.1
h_src = h_src / h_src.norm(dim=-1, keepdim=True)
scores = (h_src @ patterns.T) * 0.5
attn = torch.softmax(scores, dim=-1)
y_att = h_src + 0.1 * (attn @ patterns - h_src)
K = torch.linalg.lstsq(h_src, y_att, rcond=None).solution.T
print(f"  K: {K.shape}")


# ── 测试: attractor 拉回 → K 维持 → 测 cos 下降 ──
bs = 16
seq_len = 256
torch.manual_seed(42)
tokens = torch.randint(0, V, (bs, seq_len), device=device)
emb = model.embed(tokens)

# 直接从 pattern 中心出发（完美 on-manifold）
idx = torch.randint(0, patterns.shape[0], (bs,), device=device)
h_start = patterns[idx].clone()
h_start = h_start / h_start.norm(dim=-1, keepdim=True) * 5.4
print(f"\n  Start from pattern center. basin_cos = {basin_cos(h_start):.4f}")

# ── 对照 A: 纯 K 迭代 (无 gate) ──
print("\n── Pure K iteration (no gate) ──")
h_tmp = h_start.clone()
for step in range(101):
    if step == 0:
        print(f"  step {step:3d}: cos={basin_cos(h_tmp):.4f}")
    else:
        h_tmp = h_tmp @ K.T
        h_tmp = h_tmp / h_tmp.norm(dim=-1, keepdim=True) * 5.4
        if step in [1, 2, 5, 10, 20, 50, 100]:
            print(f"  step {step:3d}: cos={basin_cos(h_tmp):.4f}")

# ── 对照 B: gate only from pattern center ──
print("\n── Gate-only perturbation from pattern center ──")
h_tmp = h_start.clone()
for step in range(7):
    print(f"  step {step}: cos={basin_cos(h_tmp):.4f}")
    h_tmp = gate_func(h_tmp, emb[:, step % seq_len, :])
    h_tmp = h_tmp / h_tmp.norm(dim=-1, keepdim=True) * 5.4

# ── 对照 C: K@h → gate (K 预旋后 gate) ──
print("\n── K-pre rotate → gate from pattern center ──")
h_tmp = h_start.clone()
for step in range(7):
    print(f"  step {step}: cos={basin_cos(h_tmp):.4f}")
    h_pre = h_tmp @ K.T
    h_tmp = gate_func(h_pre, emb[:, step % seq_len, :])
    h_tmp = h_tmp / h_tmp.norm(dim=-1, keepdim=True) * 5.4

# ── 维持测试: attractor 拉回 → K 维持 ──
# 先跑一次 full attractor 确保在 basin 中心
h = h_start.clone()
h = gate_func(h, emb[:, 0, :])
h = F.layer_norm(attractor_func(h), [dm], Wn, bn, eps=1e-5)
h = h / h.norm(dim=-1, keepdim=True) * 5.4
print(f"\n  After 1 full attractor: cos={basin_cos(h):.4f}")

# ── 维持测试：从 t=4 开始，连续用 K 维持，记录 cos 轨迹 ──
thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]
steps_to_threshold = {t: [] for t in thresholds}

cos_trajectories = []
start_t = 4
for b in range(bs):
    h_b = h[b:b+1].clone()
    cos_history = [basin_cos(h_b)]
    for t in range(start_t, seq_len):
        x = emb[b:b+1, t, :]
        h_pre = h_b @ K.T
        h_b = gate_func(h_pre, x)
        cos_history.append(basin_cos(h_b))
    cos_trajectories.append(cos_history)

# 每条轨迹单独分析
for b in range(bs):
    traj = cos_trajectories[b]
    for th in thresholds:
        step = next((i for i, c in enumerate(traj) if c < th), len(traj))
        steps_to_threshold[th].append(step)

print(f"\n── K 维持: attractor 拉回后 cos 下降到各阈值所需步数 ──")
print(f"{'Threshold':>10} {'Median steps':>14} {'P10':>10} {'P90':>10} {'Min':>10}")
print("-" * 58)
for th in thresholds:
    vals = steps_to_threshold[th]
    sv = sorted(vals)
    print(f"{th:10.2f} {statistics.median(vals):14.1f} {sv[len(sv)//10]:10d} {sv[len(sv)*9//10]:10d} {min(vals):10d}")

# 平均 cos 轨迹
avg_traj = [statistics.mean([cos_trajectories[b][s] for b in range(bs)]) for s in range(len(cos_trajectories[0]))]
print(f"\n── 平均 cos 轨迹 (bs={bs} 样本平均) ──")
for step in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 248]:
    if step < len(avg_traj):
        print(f"  step {step:3d}: cos={avg_traj[step]:.4f}")

# ── 对照: 不用 K, 直接用 gate(h, x) (无 attractor) ──
print(f"\n── 对照: 无 attractor (gate(h, x) only) ──")
h_none = h.clone()
cos_none = [basin_cos(h_none)]
for t in range(start_t, seq_len):
    x = emb[:, t, :]
    h_none = gate_func(h_none, x)
    cos_none.append(basin_cos(h_none))
for step in [0, 1, 2, 3, 5]:
    print(f"  step {step:3d}: cos={cos_none[step]:.4f}")
print(f"  step {len(cos_none)-1:3d}: cos={cos_none[-1]:.4f}")

# ── 结论 ──
med_to_09 = statistics.median(steps_to_threshold[0.90])
if med_to_09 >= 10:
    verdict = f"CLEAN WIN — K maintains {med_to_09:.0f}+ steps, attractor call frequency drops ~{med_to_09:.0f}×"
elif med_to_09 >= 4:
    verdict = f"MODERATE — K maintains {med_to_09:.0f} steps, marginal but usable"
elif med_to_09 >= 2:
    verdict = f"WEAK — K maintains {med_to_09:.0f} steps, barely worth it"
else:
    verdict = f"DEAD — K can't even maintain 1 step"
print(f"\n  Verdict: {verdict}")
