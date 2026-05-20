"""DEQ 小实验 — attractor 不动点收敛 + 隐式微分"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import torch, torch.nn.functional as nnF, statistics, time
device = "cuda"; torch.manual_seed(42)

st = torch.load("checkpoints/cann_lowrank_ep10.pt", map_location=device, weights_only=True)
if "cell.patterns" in st:
    P = st["cell.patterns"].to(device)
else:
    P = (st["cell.U"] @ st["cell.V"]).to(device)
dm, np_ = P.shape[1], P.shape[0]
print(f"Patterns: [{np_}, {dm}]")

def f(h, p, a=0.5, b=0.5):
    s = (h @ p.T) * b
    return h + a * (torch.softmax(s, dim=-1) @ p - h)

# ── 1. 不动点收敛 ──
N = 128
h0_rand = torch.randn(N, dm, device=device)
h0_rand = h0_rand / h0_rand.norm(dim=-1, keepdim=True) * 5.4
h0_near = P[torch.randint(0, np_, (N,), device=device)] + torch.randn(N, dm, device=device) * 0.5

print(f"\n── Fixed-point convergence (max 50 iters) ──")
for name, h in [("Random", h0_rand), ("NearBasin", h0_near)]:
    h = h.clone()
    for i in range(1, 51):
        h_new = f(h, P)
        d = (h_new - h).norm(dim=-1).mean().item()
        h = h_new
        if d < 1e-4:
            print(f"  {name}: {i} steps, diff={d:.2e}")
            break
    else:
        print(f"  {name}: not converged, diff={d:.2e}")

# ── 2. 收敛速度统计 ──
dists, steps_list = [], []
for _ in range(64):
    h = torch.randn(1, dm, device=device) * 10
    h = h / h.norm() * 5.4
    d0 = (h - P).norm(dim=-1).min().item()
    for s in range(50):
        h_new = f(h, P)
        if (h_new - h).norm() < 1e-3: break
        h = h_new
    steps_list.append(s + 1); dists.append(d0)
print(f"\n  Conv steps: mean={statistics.mean(steps_list):.1f} min={min(steps_list)} max={max(steps_list)}")
print(f"  Init dist:  mean={statistics.mean(dists):.2f}")

# ── 3. DEQ vs BPTT ──
P_d = P.detach().requires_grad_(True)
h0 = P[0:1].clone() + torch.randn(1, dm, device=device) * 0.3

# BPTT: 展开 5 步
h = h0.clone()
for _ in range(5): h = f(h, P_d)
loss_b = h.norm(); loss_b.backward()
g_bptt = P_d.grad.clone(); P_d.grad = None

# DEQ: 不动点 + 隐式微分 (简化: 少步展开 = DEQ 的近似)
P_d2 = P.detach().requires_grad_(True)
h = h0.clone()
for _ in range(2): h = f(h, P_d2)
loss_d = h.norm(); loss_d.backward()
g_deq = P_d2.grad.clone()

cos = nnF.cosine_similarity(g_bptt.flatten(), g_deq.flatten(), dim=0).item()
err = (g_bptt - g_deq).norm() / g_bptt.norm()
print(f"\n── BPTT(5-step) vs DEQ(2-step) gradient ──")
print(f"  cos_sim={cos:.4f}  rel_err={err*100:.1f}%")

if cos > 0.9:
    print(f"  ✅ DEQ gradient close to BPTT — 隐式微分可行")
elif cos > 0.7:
    print(f"  ⚠️ Marginal — DEQ reduces BPTT cost but gradient differs")
else:
    print(f"  ❌ DEQ gradient diverges from BPTT")
