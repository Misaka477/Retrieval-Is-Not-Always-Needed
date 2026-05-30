"""
线性 K attractor 自回归误差传播测试。

核心问题:
  自回归生成时，每一步的微小误差会不会让状态飘出 manifold？
  非线性 attractor 是 contraction (α<1 → 永远拉回 basin)
  线性 K 是松开的——它学的是 manifold 上的行为，偏离后可能不准

测量:
  1. ||h_t|| 轨迹 (norm 发散 = 误差累积)
  2. min_i ||h_t - P[i]|| (离最近 basin 的距离)
  3. 引入周期性 "snap-back" 的修复效果
"""
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_patterns(ckpt_path):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "cell.patterns" in st:
        return st["cell.patterns"].to(device)
    if "cell.U" in st and "cell.V" in st:
        return (st["cell.U"] @ st["cell.V"]).to(device)
    raise KeyError(f"No patterns in {ckpt_path}")


def attractor_nonlinear(h, patterns, beta=0.5, alpha=0.1):
    scores = (h @ patterns.T) * beta
    attn = torch.softmax(scores, dim=-1)
    attracted = attn @ patterns
    return h + alpha * (attracted - h)


def fit_K(patterns, beta=0.5, alpha=0.1, n_samples=8192):
    np_, dm = patterns.shape
    torch.manual_seed(42)
    idx = torch.randint(0, np_, (n_samples,), device=device)
    h_src = patterns[idx] + torch.randn(n_samples, dm, device=device) * 0.1
    h_src = h_src / h_src.norm(dim=-1, keepdim=True)
    y = attractor_nonlinear(h_src, patterns, beta=beta, alpha=alpha)
    K = torch.linalg.lstsq(h_src, y, rcond=None).solution.T
    return K


def attractor_linear(h, K, alpha=0.1):
    return h + alpha * (h @ K.T - h)


def dist_to_nearest_basin(h, patterns):
    """每个 batch 元素到最近 pattern 的 L2 距离."""
    # h: [batch, dm], patterns: [np, dm]
    h_n = F.normalize(h, dim=-1)
    p_n = F.normalize(patterns, dim=-1)
    sim = h_n @ p_n.T
    max_sim, _ = sim.max(dim=-1)
    return max_sim  # cos_sim, 1=perfect match


def rollout(model_attractor, h_init, steps, patterns, inject_noise=True, noise_scale=0.02):
    """自回归 rollout 记录状态轨迹."""
    bs, dm = h_init.shape
    h = h_init.clone()
    norms = []
    basin_dists = []

    for t in range(steps):
        # 模拟 gate 输出: h_ssm ≈ h + 噪声 (mock SSM gate)
        if inject_noise:
            h_ssm = h + torch.randn(bs, dm, device=device) * noise_scale
        else:
            h_ssm = h
        h_ssm = h_ssm / h_ssm.norm(dim=-1, keepdim=True)

        # attractor
        h_new = model_attractor(h_ssm)

        norms.append(h_new.norm(dim=-1).mean().item())
        basin_dists.append(dist_to_nearest_basin(h_new, patterns).mean().item())

        h = h_new

    return norms, basin_dists


def rollout_with_snap(h_init, steps, K, patterns, beta=0.5, alpha=0.1, snap_every=20):
    """带周期性 snap-back 的 rollout: 每 snap_every 步拉回最近 basin."""
    bs, dm = h_init.shape
    h = h_init.clone()
    norms = []
    basin_dists = []

    for t in range(steps):
        h_ssm = h + torch.randn(bs, dm, device=device) * 0.02
        h_ssm = h_ssm / h_ssm.norm(dim=-1, keepdim=True)

        if t > 0 and t % snap_every == 0:
            scores = (h @ patterns.T) * beta
            idx = scores.argmax(dim=-1)
            h = h + alpha * (patterns[idx] - h)

        h_new = attractor_linear(h_ssm, K, alpha=alpha)
        norms.append(h_new.norm(dim=-1).mean().item())
        basin_dists.append(dist_to_nearest_basin(h_new, patterns).mean().item())
        h = h_new

    return norms, basin_dists


def main():
    CKPT = os.path.join(_ROOT, "checkpoints", "cann_lowrank_ep10.pt")
    if not os.path.exists(CKPT):
        CKPT = os.path.join(_ROOT, "checkpoints", "cann_lowrank_final.pt")
    print(f"Loading: {CKPT}")
    patterns = load_patterns(CKPT)
    dm = patterns.shape[1]
    print(f"  patterns: {patterns.shape}")

    K = fit_K(patterns)
    print(f"  K fitted: {K.shape}\n")

    bs = 8
    steps = 8192
    noise_scale = 0.05  # 模拟 gate 扰动
    torch.manual_seed(42)

    # 从随机 pattern 附近出发
    idx = torch.randint(0, patterns.shape[0], (bs,), device=device)
    h_init = patterns[idx] + torch.randn(bs, dm, device=device) * 0.05
    h_init = h_init / h_init.norm(dim=-1, keepdim=True)

    def make_attractor_nonlinear():
        return lambda h: attractor_nonlinear(h, patterns, beta=0.5, alpha=0.1)
    
    def make_attractor_linear():
        return lambda h: attractor_linear(h, K, alpha=0.1)
    
    def make_attractor_none():
        return lambda h: h

    configs = [
        ("Nonlinear",    make_attractor_nonlinear()),
        ("Linear K",     make_attractor_linear()),
        ("None",         make_attractor_none()),
    ]

    print(f"Rollout: {steps} steps, bs={bs}, noise_scale={noise_scale}\n")
    all_results = {}
    for name, attractor_fn in configs:
        print(f"  {name}...")
        norms, basin_dists = rollout(attractor_fn, h_init, steps, patterns,
                                     inject_noise=True, noise_scale=noise_scale)
        all_results[name] = (norms, basin_dists)

    # ── 噪声阈值扫描：找到线性 K 的稳定边界 ──
    print(f"\n{'='*72}")
    print("Noise threshold scan: finding linear K stability boundary")
    print(f"{'='*72}")
    print(f"{'Noise':>8} {'LinK final':>12} {'NL final':>12} {'Verdict':>20}")
    print("-" * 56)
    h0 = h_init
    noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]
    for ns in noise_levels:
        _, blin = rollout(make_attractor_linear(), h0, 256, patterns, noise_scale=ns)
        _, bnl = rollout(make_attractor_nonlinear(), h0, 256, patterns, noise_scale=ns)
        avg_end_lin = sum(blin[-64:]) / 64
        avg_end_nl = sum(bnl[-64:]) / 64
        status = "STABLE" if avg_end_lin > 0.8 else "COLLAPSED" if avg_end_lin < 0.3 else "MID"
        print(f"{ns:8.2f} {avg_end_lin:12.4f} {avg_end_nl:12.4f} {status:>20}")
    
    print(f"\n{'='*72}")
    print(f"Extreme Rollout: {steps} steps with noise={noise_scale}")
    print(f"{'='*72}")

    checkpoints = [0, 7, 31, 127, 511, 1023, 2047, 4095, 8191]
    print(f"\n── Basin cos_sim (1=on basin, ↓=drifting) ──")
    header = f"{'Step':>6}"
    for name, _ in configs:
        header += f" {'':>3}{name:>12}"
    print(header)
    print("-" * (8 + 16 * len(configs)))
    for s in checkpoints:
        sn = min(s, steps - 1)
        row = f"{sn+1:6d}"
        for name, _ in configs:
            bd = all_results[name][1][sn]
            row += f" {bd:12.4f}"
        print(row)

    # 漂移速度
    print(f"\n── Drift velocity (cos_sim/step × 1e6) ──")
    for name, _ in configs:
        bd = torch.tensor(all_results[name][1])
        # linear fit slope
        t = torch.arange(len(bd), dtype=torch.float32)
        slope = (t * (bd - bd.mean())).sum() / (t * (t - t.mean())).sum()
        print(f"  {name:12s}: {slope*1e6:+.2f} per step  "
              f"({'stable' if abs(slope) < 1e-6 else 'drifting' if slope < 0 else 'improving'})")

    # 最终稳定值 vs 最初
    print(f"\n── Final cos_sim ──")
    for name, _ in configs:
        bd = all_results[name][1]
        avg_end = sum(bd[-256:]) / 256
        avg_start = sum(bd[:64]) / 64
        print(f"  {name:12s}: {avg_end:.4f}  (early: {avg_start:.4f}, Δ={avg_end-avg_start:+.4f})")

    # 最差样本
    print(f"\n── Worst sample at final step (cos_sim) ──")
    # we need per-sample tracking...let me compute final separately
    # Since rollout averages across batch, let's do single-sample worst case
    idx0 = idx[0]
    h_single = h_init[0:1].clone()
    for t in range(steps):
        h_single = h_single + torch.randn(1, dm, device=device) * noise_scale
        h_single = h_single / h_single.norm(dim=-1, keepdim=True)
        h_single = attractor_linear(h_single, K, alpha=0.1)
    worst_lin = dist_to_nearest_basin(h_single, patterns).item()
    
    h_single = h_init[0:1].clone()
    for t in range(steps):
        h_single = h_single + torch.randn(1, dm, device=device) * noise_scale
        h_single = h_single / h_single.norm(dim=-1, keepdim=True)
        h_single = attractor_nonlinear(h_single, patterns, beta=0.5, alpha=0.1)
    worst_nl = dist_to_nearest_basin(h_single, patterns).item()
    
    h_single = h_init[0:1].clone()
    for t in range(steps):
        h_single = h_single + torch.randn(1, dm, device=device) * noise_scale
        h_single = h_single / h_single.norm(dim=-1, keepdim=True)
    worst_none = dist_to_nearest_basin(h_single, patterns).item()
    
    print(f"  Nonlinear: {worst_nl:.4f}")
    print(f"  Linear K:  {worst_lin:.4f}")
    print(f"  None:      {worst_none:.4f}")


if __name__ == "__main__":
    main()
