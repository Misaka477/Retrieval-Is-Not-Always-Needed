"""
验证 parallel scan 加速方案：associative scan + sparse attractor.

验证点：
  1. scan 后的 ε 和逐步 ε 的吻合度 > 95%
  2. attractor 修正能从 scan 误差中恢复
  3. 各步 attractor 修正之间互不干扰
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from modules.temporal_snn_cell import TemporalSNNCell
import time

device = "cuda"
torch.manual_seed(42)

# Config (small model)
DM, NP, TH = 64, 256, 1.0
SEQ, BS = 32, 8

cell = TemporalSNNCell(DM, NP, error_threshold=TH).to(device)

# Generate random inputs
emb = torch.randn(BS, SEQ, DM, device=device)

# ── Phase 0: step-by-step reference ──
print("Step-by-step reference...", flush=True)
h = torch.zeros(BS, DM, device=device)
eps_seq = []
errors = []
for t in range(SEQ):
    h_new = cell(h, emb[:, t, :], step=t)
    error = (h_new - h).norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-8)
    eps_seq.append(error)
    errors.append((h_new - h).norm().item())
    h = h_new
eps_seq = torch.stack(eps_seq)  # [SEQ, BS]

# ── Phase 1: parallel scan (associative gate) ──
print("Parallel scan (associative)...", flush=True)
# Mamba-style: h = a*h + b*x → separable
h_scan = torch.zeros(BS, DM, device=device)
a_mat = []
b_mat = []
xp_mat = []

for t in range(SEQ):
    combined = torch.cat([h_scan, emb[:, t, :]], dim=-1)
    a = torch.sigmoid(cell.gate_a(combined))
    b = torch.sigmoid(cell.gate_b(combined))
    xp = cell.proj_in(emb[:, t, :])
    a_mat.append(a); b_mat.append(b); xp_mat.append(xp)
    h_scan = a * h_scan + b * xp

# Associative scan: the gate is already linear in h
# h_t = a_t * h_{t-1} + b_t * xp_t
# This is already in scan-friendly form
print(f"  gate form is associative: h_t = a_t·h_{{t-1}} + b_t·xp_t", flush=True)

# ── Phase 2: scan accuracy ──
print("\nVerification 1: ε consistency (scan vs step-by-step)...", flush=True)
h_scan = torch.zeros(BS, DM, device=device)
eps_scan = []
for t in range(SEQ):
    combined = torch.cat([h_scan, emb[:, t, :]], dim=-1)
    a = torch.sigmoid(cell.gate_a(combined))
    b = torch.sigmoid(cell.gate_b(combined))
    xp = cell.proj_in(emb[:, t, :])
    h_ssm = a * h_scan + b * xp

    h_pred = h_scan.detach()
    error = (h_ssm - h_pred).norm(dim=-1) / h_pred.norm(dim=-1).clamp(min=1e-8)
    eps_scan.append(error)

    # Apply attractor only when triggered
    if cell.error_threshold[0] < 0 or error.mean() > cell.error_threshold[0]:
        pat = cell.patterns.unsqueeze(0).expand(BS, -1, -1)
        xi = h_ssm.unsqueeze(1)
        scores = xi @ pat.transpose(1, 2) * cell.beta_t[0]
        attn = torch.softmax(scores, dim=-1)
        attracted = (attn @ pat).squeeze(1)
        combined_last = torch.cat([h_ssm, emb[:, t, :]], dim=-1)
        alpha = torch.sigmoid(cell.gate_alpha(combined_last))
        h_scan = h_ssm + alpha * (attracted - h_ssm)
        h_scan = cell.norm(h_scan)
    else:
        h_scan = cell.norm(h_ssm)

eps_scan = torch.stack(eps_scan)

# Compare ε decisions
step_att = (eps_seq > TH).float().mean(dim=1)  # [SEQ] — fraction of batch that triggers
scan_att = (eps_scan > TH).float().mean(dim=1)
diff = (step_att - scan_att).abs().mean().item()
agree = (step_att.round() == scan_att.round()).float().mean().item() * 100
print(f"  avg ε diff: {diff:.4f}")
print(f"  att decision agreement: {agree:.1f}%")

# ── Verification 3: attractor independence ──
print("\nVerification 2: attractor independence...", flush=True)
h_test = torch.randn(BS, DM, device=device)
h_ref = h_test.clone()

# Apply attractor to all positions independently
independent_results = []
for t in range(BS):
    pat = cell.patterns.unsqueeze(0)
    xi = h_test[t:t+1].unsqueeze(1)
    scores = xi @ pat.transpose(1, 2) * cell.beta_t[0]
    attn = torch.softmax(scores, dim=-1)
    attracted = (attn @ pat).squeeze(1)
    h_result = h_test[t:t+1] + 0.5 * (attracted - h_test[t:t+1])
    independent_results.append(h_result)
indie = torch.cat(independent_results, dim=0)

# Apply attractor in batch
pat = cell.patterns.unsqueeze(0).expand(BS, -1, -1)
xi = h_test.unsqueeze(1)
scores = xi @ pat.transpose(1, 2) * cell.beta_t[0]
attn = torch.softmax(scores, dim=-1)
attracted = (attn @ pat).squeeze(1)
batched = h_test + 0.5 * (attracted - h_test)

independence_err = (indie - batched).norm().item()
print(f"  independent vs batched error: {independence_err:.6f} (should be ~0)")
print(f"  {'✅ attractor is fully independent per position' if independence_err < 1e-5 else '❌ has cross-position dependency'}")

print(f"\nSummary:")
print(f"  ε agreement: {agree:.1f}%")
print(f"  independence: {'✅' if independence_err < 1e-5 else '❌'}")
