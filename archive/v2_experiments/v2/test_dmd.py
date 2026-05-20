"""
DMD / Koopman 线性化 — 将 CANN attractor 非线性映射拟合为线性算子。

核心思路:
  attractor 步: h_new = h + alpha * (softmax(h @ P^T) @ P - h)
  目标:          找到线性算子 K，使得 K @ h ≈ f_attractor(h)
  
  推理时: h_new = K @ h  (单次 matmul, 无 softmax, 无 pattern 存储)
  K 低秩分解后: O(r*dm) vs 原 O(np*dm)

实验步骤:
  1. 加载训练好的 patterns 矩阵
  2. 采样 h_ssm 状态 (Gaussian + 模型实际分布)
  3. 计算精确 attractor 输出 → 构造 (X, Y) 对
  4. 最小二乘拟合 K = Y @ X^†
  5. 测量重构质量 vs rank
  6. 速度对比: 线性 K vs 非线性 softmax-attractor
"""
import sys, os, time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


def load_patterns(ckpt_path):
    """从 checkpoint 提取 effective patterns 矩阵."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "cell.patterns" in st:
        patterns = st["cell.patterns"]
    elif "cell.U" in st and "cell.V" in st:
        patterns = st["cell.U"] @ st["cell.V"]
    else:
        raise KeyError(f"No patterns/U/V found in {ckpt_path}. Keys: {list(st.keys())[:20]}")
    dm = patterns.shape[1]
    np = patterns.shape[0]
    return patterns, dm, np


def attractor_exact(h, patterns, beta=1.0, alpha=0.1):
    """精确非线性 attractor 前向."""
    scores = (h @ patterns.T) * beta
    attn = torch.softmax(scores, dim=-1)
    attracted = attn @ patterns
    return h + alpha * (attracted - h)


def fit_linear_operator(X, Y, ridge=1e-6):
    """
    X: [N, dm] — 输入 h_ssm 状态
    Y: [N, dm] — 精确 attractor 输出 h_new
    Returns K: [dm, dm] — 最小二乘线性算子
    """
    K = torch.linalg.lstsq(
        X, Y, rcond=None
    ).solution.T
    if ridge > 0:
        I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype) * ridge
        K_ridge = torch.linalg.solve(
            X.T @ X + I, X.T @ Y
        ).T
        return K, K_ridge
    return K, None


def svd_error_vs_rank(K, X, Y, ranks):
    """K 的低秩近似在不同 rank 下的重构误差."""
    U, S, Vt = torch.linalg.svd(K, full_matrices=False)
    errors = {}
    for r in ranks:
        Kr = (U[:, :r] * S[:r]) @ Vt[:r, :]
        Y_hat = X @ Kr.T
        err = (Y - Y_hat).norm() / Y.norm()
        errors[r] = err.item()
    return errors


def main():
    CKPT = os.path.join(_ROOT, "checkpoints", "cann_lowrank_ep10.pt")
    if not os.path.exists(CKPT):
        CKPT = os.path.join(_ROOT, "checkpoints", "cann_lowrank_final.pt")
    print(f"Loading: {CKPT}")
    patterns, dm, np = load_patterns(CKPT)
    print(f"  d_model={dm}, n_patterns={np}")
    patterns = patterns.to(device)

    # ── 1. 采样状态 ──
    N = 8192
    print(f"\nSampling {N} states...")
    torch.manual_seed(42)

    h_gauss = torch.randn(N, dm, device=device)
    h_gauss = h_gauss / h_gauss.norm(dim=-1, keepdim=True)
    y_gauss = attractor_exact(h_gauss, patterns, beta=0.5, alpha=0.1)

    # 从 pattern space 采 (更接近实际激活分布)
    idx = torch.randint(0, np, (N,), device=device)
    h_pattern = patterns[idx] + torch.randn(N, dm, device=device) * 0.1
    h_pattern = h_pattern / h_pattern.norm(dim=-1, keepdim=True)
    y_pattern = attractor_exact(h_pattern, patterns, beta=0.5, alpha=0.1)

    # ── 2. 拟合线性算子 ──
    print("Fitting linear operator K...")
    t0 = time.time()

    K_gauss, K_gauss_ridge = fit_linear_operator(h_gauss, y_gauss)
    K_patt, K_patt_ridge = fit_linear_operator(h_pattern, y_pattern)

    print(f"  Fit time: {time.time()-t0:.2f}s")

    # ── 3. 重构质量 ──
    print("\n── Gaussian-dist states ──")
    y_hat = h_gauss @ K_gauss.T
    err = (y_gauss - y_hat).norm() / y_gauss.norm()
    cos = F.cosine_similarity(y_hat, y_gauss, dim=-1).mean()
    print(f"  OLS K:   rel_err={err*100:.1f}%  cos_sim={cos:.4f}")

    if K_gauss_ridge is not None:
        y_hat_r = h_gauss @ K_gauss_ridge.T
        err_r = (y_gauss - y_hat_r).norm() / y_gauss.norm()
        cos_r = F.cosine_similarity(y_hat_r, y_gauss, dim=-1).mean()
        print(f"  Ridge K: rel_err={err_r*100:.1f}%  cos_sim={cos_r:.4f}")

    print("\n── Pattern-perturbed states ──")
    y_hat = h_pattern @ K_patt.T
    err = (y_pattern - y_hat).norm() / y_pattern.norm()
    cos = F.cosine_similarity(y_hat, y_pattern, dim=-1).mean()
    print(f"  OLS K:   rel_err={err*100:.1f}%  cos_sim={cos:.4f}")

    if K_patt_ridge is not None:
        y_hat_r = h_pattern @ K_patt_ridge.T
        err_r = (y_pattern - y_hat_r).norm() / y_pattern.norm()
        cos_r = F.cosine_similarity(y_hat_r, y_pattern, dim=-1).mean()
        print(f"  Ridge K: rel_err={err_r*100:.1f}%  cos_sim={cos_r:.4f}")

    # ── 4. low-rank SVD 误差 ──
    print(f"\n── SVD rank vs error (pattern-perturbed states) ──")
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, -1]
    for r in ranks:
        if r == -1:
            r_eff = dm
            label = "full"
        else:
            r_eff = r
            label = f"r={r}"
        U, S, Vt = torch.linalg.svd(K_patt, full_matrices=False)
        Kr = (U[:, :r_eff] * S[:r_eff]) @ Vt[:r_eff, :]
        y_hat = h_pattern @ Kr.T
        err = (y_pattern - y_hat).norm() / y_pattern.norm()
        cos = F.cosine_similarity(y_hat, y_pattern, dim=-1).mean()
        print(f"  {label:6s}: rel_err={err*100:5.1f}%  cos_sim={cos:.4f}")

    # ── 5. 速度对比 ──
    print(f"\n── Speed (batch=8, dm={dm}, np={np}) ──")
    bs = 8
    h_test = torch.randn(bs, dm, device=device)
    warmup, reps = 10, 1000

    # 非线性 attractor
    for _ in range(warmup):
        attractor_exact(h_test, patterns, beta=0.5, alpha=0.1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        attractor_exact(h_test, patterns, beta=0.5, alpha=0.1)
    torch.cuda.synchronize()
    t_nonlinear = (time.perf_counter() - t0) / reps

    # 线性 K
    for _ in range(warmup):
        h_test @ K_patt.T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        h_test @ K_patt.T
    torch.cuda.synchronize()
    t_linear = (time.perf_counter() - t0) / reps

    # 低秩线性 K (r=128): K_r = U_r @ diag(S_r) @ V_r
    r = 128
    U_r_T = U[:, :r].T                                   # [r, dm]
    S_r = torch.diag(S[:r])                              # [r, r]
    V_r_T = Vt[:r, :].T                                  # [dm, r]
    K_lr_T = V_r_T @ S_r @ U_r_T                         # [dm, dm]
    for _ in range(warmup):
        h_test @ K_lr_T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        h_test @ K_lr_T
    torch.cuda.synchronize()
    t_lowrank = (time.perf_counter() - t0) / reps

    print(f"  Nonlinear  (softmax+matmul): {t_nonlinear*1e6:8.1f} us")
    print(f"  Linear K   (full [dm,dm]):  {t_linear*1e6:8.1f} us ({t_nonlinear/t_linear:.1f}x)")
    print(f"  Linear Kr  (r=128):          {t_lowrank*1e6:8.1f} us ({t_nonlinear/t_lowrank:.1f}x)")

    # ── 6. 与 identity baseline 对比 ──
    print(f"\n── vs h_ssm 直接输出 (alpha=0, no attractor) ──")
    err_identity = (y_pattern - h_pattern).norm() / y_pattern.norm()
    cos_identity = F.cosine_similarity(h_pattern, y_pattern, dim=-1).mean()
    print(f"  Identity: rel_err={err_identity*100:.1f}%  cos_sim={cos_identity:.4f}")


if __name__ == "__main__":
    main()
