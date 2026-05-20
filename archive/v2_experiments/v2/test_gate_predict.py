"""
Gate 预判测试 — 能否用轻量线性层预测 gate_a 的输出？

思路:
  当前 gate_a = sigmoid(combined @ Wa + ba)   [batch, 2*dm] @ [2*dm, dm]
  预判 gate_a ≈ sigmoid(combined @ W_light)   [batch, 2*dm] @ [2*dm, 1] (per-dim)
  
  如果预判够准 → 硬遗忘维度 (a<0.1) 跳过 gate 计算
                 硬保持维度 (a>0.75) 跳过 gate 计算
                 中间维度 → 正常 gate 计算

测试:
  1. 收集真实 forward 的 (combined, gate_a) 对
  2. 拟合 per-dimension 轻量预测器
  3. 测 recall/precision: 预判为"可跳过"的维度中，实际可跳过的比例
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
st = torch.load(os.path.join(_ROOT, CKPT), map_location=device, weights_only=True)
model = RINASeqModel(V, d_model=dm, n_patterns=4096, beta=0.5, n_slots=V,
                     attract_every=2, pattern_rank=128).to(device)
model.load_state_dict(st, strict=False)
model.eval()

Wa = model.cell.gate_a.weight.detach()
ba = model.cell.gate_a.bias.detach()

# ── 收集数据 ──
seq = torch.randint(0, V, (16, 128), device=device)
emb = model.embed(seq)
h = torch.zeros(16, dm, device=device)

all_combined = []
all_gate_a = []

for t in range(127):
    combined = torch.cat([h, emb[:, t, :]], dim=-1)
    a = torch.sigmoid(combined @ Wa.T + ba)
    all_combined.append(combined.detach())
    all_gate_a.append(a.detach())
    h = model.cell(h, emb[:, t, :], step=t)

X = torch.cat(all_combined, dim=0)  # [N, 2*dm]
Y = torch.cat(all_gate_a, dim=0)    # [N, dm]
N, D = X.shape
dm2 = D // 2
print(f"Collected: {N} samples, 2*dm={D}, dm={dm2}")

# ── 拟合 per-dimension 轻量预测器 ──
# 为每个维度独立拟合: gate_a[d] ≈ sigmoid(combined @ w_d)
# w_d shape: [2*dm] → 参数量 dm * 2*dm (和 Wa 一样大)
# 但可以共享大部分维度，做低秩版本

# 方案: 全局线性 scorce_i + 维度专用 bias
# gate_a[d] ≈ sigmoid(combined @ W_shared[d,:] + b_d)

# 最简: 逐维度逻辑回归
split = int(N * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Per-dim logistic regression via least squares on logit
# logit(a) = combined @ w_d → log(a/(1-a)) ≈ combined @ w_d
eps = 1e-6
logit_Y = torch.log((Y_train + eps) / (1 - Y_train + eps))

W_pred = torch.zeros(dm2, D, device=device)
for d in range(dm2):
    W_pred[d] = torch.linalg.lstsq(X_train, logit_Y[:, d].unsqueeze(1),
                                    rcond=None).solution.squeeze()

print(f"Fitted {dm2} per-dim predictors")

# ── 评估 ──
logit_pred = X_test @ W_pred.T
Y_pred = torch.sigmoid(logit_pred)
mse = (Y_test - Y_pred).pow(2).mean().item()
corr = F.cosine_similarity(Y_test.flatten().unsqueeze(0),
                           Y_pred.flatten().unsqueeze(0)).item()
print(f"\nMSE={mse:.4f}  cos_sim={corr:.4f}")

# ── 跳过逻辑: 预判阈值 ──
# 预判为 <0.1(可跳过gate,直接置0) 或 >0.75(可跳过gate,保持h)
skip_low_thresh = 0.1
skip_high_thresh = 0.75

# 真实可跳过的维度
true_skip_low = (Y_test < skip_low_thresh).float()
true_skip_high = (Y_test > skip_high_thresh).float()
true_skip = true_skip_low + true_skip_high

# 预判可跳过的维度
pred_skip_low = (Y_pred < skip_low_thresh).float()
pred_skip_high = (Y_pred > skip_high_thresh).float()
pred_skip = pred_skip_low + pred_skip_high

# 统计
total = true_skip.numel()
true_skip_frac = true_skip.mean().item()
pred_skip_frac = pred_skip.mean().item()

# Precision: 预判可跳过中，实际可跳过的比例
# 预判 low, 实际也是 low
tp_low = (pred_skip_low * true_skip_low).sum().item()
fp_low = (pred_skip_low * (1 - true_skip_low)).sum().item()
tp_high = (pred_skip_high * true_skip_high).sum().item()
fp_high = (pred_skip_high * (1 - true_skip_high)).sum().item()

precision_low = tp_low / (tp_low + fp_low) if (tp_low + fp_low) > 0 else 0
precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 0

# Recall: 实际可跳过中，预判对的比例
fn_low = ((1 - pred_skip_low) * true_skip_low).sum().item()
fn_high = ((1 - pred_skip_high) * true_skip_high).sum().item()
recall_low = tp_low / (tp_low + fn_low) if (tp_low + fn_low) > 0 else 0
recall_high = tp_high / (tp_high + fn_high) if (tp_high + fn_high) > 0 else 0

print(f"\n── Skip prediction quality ──")
print(f"  True skip fraction:    {true_skip_frac*100:.1f}%")
print(f"  Pred skip fraction:    {pred_skip_frac*100:.1f}%")
print(f"  Skip low (a<0.1):     precision={precision_low*100:.1f}%  recall={recall_low*100:.1f}%")
print(f"  Skip high (a>0.75):   precision={precision_high*100:.1f}%  recall={recall_high*100:.1f}%")

# ── 代价: 用 W_pred 的参数量 vs Wa 的参数量 ──
print(f"\n── Parameter cost ──")
print(f"  Original Wa:  {Wa.numel():,}  ({dm2}×{D})")
print(f"  Predict W:    {W_pred.numel():,}  ({dm2}×{D})  ← 一样大!")

# ── 低秩预判 ──
# 用 rank=16 的分解代替 per-dim predictor
rank = 16
U = torch.randn(dm2, rank, device=device) * 0.02
V = torch.randn(rank, D, device=device) * 0.02
U = torch.nn.Parameter(U)
V = torch.nn.Parameter(V)
opt = torch.optim.Adam([U, V], lr=1e-3)
for step in range(200):
    logit_lr = X_train @ V.T @ U.T  # [N, dm2]
    loss = (logit_lr - logit_Y).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    Y_pred_lr = torch.sigmoid(X_test @ V.T @ U.T)
mse_lr = (Y_test - Y_pred_lr).pow(2).mean().item()
corr_lr = F.cosine_similarity(Y_test.flatten().unsqueeze(0),
                               Y_pred_lr.flatten().unsqueeze(0)).item()

pred_skip_low_lr = (Y_pred_lr < skip_low_thresh).float()
pred_skip_high_lr = (Y_pred_lr > skip_high_thresh).float()
pred_skip_lr = pred_skip_low_lr + pred_skip_high_lr

tp_low_lr = (pred_skip_low_lr * true_skip_low).sum().item()
fp_low_lr = (pred_skip_low_lr * (1 - true_skip_low)).sum().item()
tp_high_lr = (pred_skip_high_lr * true_skip_high).sum().item()
fp_high_lr = (pred_skip_high_lr * (1 - true_skip_high)).sum().item()

prec_low_lr = tp_low_lr / (tp_low_lr + fp_low_lr) if (tp_low_lr + fp_low_lr) > 0 else 0
prec_high_lr = tp_high_lr / (tp_high_lr + fp_high_lr) if (tp_high_lr + fp_high_lr) > 0 else 0
fn_low_lr = ((1 - pred_skip_low_lr) * true_skip_low).sum().item()
fn_high_lr = ((1 - pred_skip_high_lr) * true_skip_high).sum().item()
rec_low_lr = tp_low_lr / (tp_low_lr + fn_low_lr) if (tp_low_lr + fn_low_lr) > 0 else 0
rec_high_lr = tp_high_lr / (tp_high_lr + fn_high_lr) if (tp_high_lr + fn_high_lr) > 0 else 0

lr_params = U.numel() + V.numel()
skip_frac_lr = pred_skip_lr.mean().item()

print(f"\n── Low-rank predictor (rank={rank}) ──")
print(f"  MSE={mse_lr:.4f}  cos_sim={corr_lr:.4f}")
print(f"  Params: {lr_params:,}  ({lr_params / Wa.numel() * 100:.1f}% of Wa)")
print(f"  Skip fraction: {skip_frac_lr*100:.1f}%")
print(f"  Skip low (a<0.1):  precision={prec_low_lr*100:.1f}%  recall={rec_low_lr*100:.1f}%")
print(f"  Skip high (a>0.75): precision={prec_high_lr*100:.1f}%  recall={rec_high_lr*100:.1f}%")

# ── 判定 ──
gate_saved = skip_frac_lr * (1 - lr_params / Wa.numel())
print(f"\n── Verdict ──")
if prec_low_lr > 0.8 and prec_high_lr > 0.8:
    print(f"  PASS: both precision > 80%.  Gate saved: {gate_saved*100:.1f}%")
elif prec_low_lr > 0.6 and prec_high_lr > 0.6:
    print(f"  MARGINAL: precision > 60%.  Gate saved: {gate_saved*100:.1f}%")
else:
    print(f"  FAIL: precision insufficient for safe skipping")

# ── Rank sweep: 找到 precision>90% 的最低 rank ──
print(f"\n── Rank sweep for precision > 90% ──")
print(f"{'Rank':>6} {'Params':>10} {'Param%':>7} {'Prec_low':>10} {'Rec_low':>10} {'Prec_high':>10} {'Rec_high':>10} {'Skip%':>7}")
print("-" * 77)
for rank in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]:
    U = torch.randn(dm2, rank, device=device) * 0.02
    V = torch.randn(rank, D, device=device) * 0.02
    U = torch.nn.Parameter(U); V = torch.nn.Parameter(V)
    opt = torch.optim.Adam([U, V], lr=1e-3)
    for step in range(200):
        logit_lr = X_train @ V.T @ U.T
        loss = (logit_lr - logit_Y).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Yp = torch.sigmoid(X_test @ V.T @ U.T)
    pl = (Yp < 0.1).float(); ph = (Yp > 0.75).float()
    tl = (Y_test < 0.1).float(); th = (Y_test > 0.75).float()
    tp_l = (pl*tl).sum().item(); fp_l = (pl*(1-tl)).sum().item()
    tp_h = (ph*th).sum().item(); fp_h = (ph*(1-th)).sum().item()
    fn_l = ((1-pl)*tl).sum().item(); fn_h = ((1-ph)*th).sum().item()
    prec_l = tp_l/(tp_l+fp_l) if (tp_l+fp_l)>0 else 0
    rec_l = tp_l/(tp_l+fn_l) if (tp_l+fn_l)>0 else 0
    prec_h = tp_h/(tp_h+fp_h) if (tp_h+fp_h)>0 else 0
    rec_h = tp_h/(tp_h+fn_h) if (tp_h+fn_h)>0 else 0
    skip_frac = (pl+ph).mean().item()
    n_param = U.numel() + V.numel()
    print(f"{rank:6d} {n_param:10,d} {n_param/Wa.numel()*100:6.1f}% "
          f"{prec_l*100:9.1f}% {rec_l*100:9.1f}% "
          f"{prec_h*100:9.1f}% {rec_h*100:9.1f}% {skip_frac*100:6.1f}%")
    if prec_l > 0.90 and prec_h > 0.90:
        print(f"  → FIRST rank with precision > 90%!")
        break
# 如果 combined 本身就和 gate_a 线性相关度高，预判才有意义
print(f"\n── Intrinsic predictability ──")
# 用单个全局 regressor
W_single = torch.linalg.lstsq(X_train, logit_Y, rcond=None).solution  # [2*dm, dm]
Y_single = torch.sigmoid(X_test @ W_single)
mse_single = (Y_test - Y_single).pow(2).mean().item()
corr_single = F.cosine_similarity(Y_test.flatten().unsqueeze(0),
                                   Y_single.flatten().unsqueeze(0)).item()
print(f"  Single global regressor:  MSE={mse_single:.4f}  cos_sim={corr_single:.4f}")
print(f"  Per-dim regressor:        MSE={mse:.4f}  cos_sim={corr:.4f}")
print(f"  Low-rank r={rank}:        MSE={mse_lr:.4f}  cos_sim={corr_lr:.4f}")

# real gate_a using sigmoid(Wa) as upper bound
Y_oracle = torch.sigmoid(X_test @ Wa.T + ba)
mse_oracle = (Y_test - Y_oracle).pow(2).mean().item()
corr_oracle = F.cosine_similarity(Y_test.flatten().unsqueeze(0),
                                   Y_oracle.flatten().unsqueeze(0)).item()
print(f"  Oracle (sigmoid(Wa)):     MSE={mse_oracle:.4f}  cos_sim={corr_oracle:.4f} ← ceiling")
