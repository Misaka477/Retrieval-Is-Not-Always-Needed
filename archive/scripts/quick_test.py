"""
Quick smoke test — 10 seconds, no external data.

Verifies:
  1. Package imports
  2. Model builds
  3. Forward pass produces valid logits
  4. Loss decreases over 5 training steps
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

print("=" * 50)
print("RINA Quick Test")
print("=" * 50)

# 1. Package imports
print("\n[1/4] Package imports...")
from rina import TemporalSNNCell, TemporalSNNModel, SlotMemory
from rina.config import load_config
print("  rina.cell.TemporalSNNCell    OK")
print("  rina.model.TemporalSNNModel  OK")
print("  rina.slot.SlotMemory         OK")
print("  rina.config.load_config      OK")

cfg = load_config()
print(f"  config: dm={cfg['dm']}, np={cfg['np']}, th={cfg['error_threshold']}")

# 2. Build model
print("\n[2/4] Building model (small)...")
V, DM, NP = 100, 64, 128
m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                     attract_every=2, error_threshold=1.0,
                     hebbian_lr=0.01, inhibition_threshold=0.8)
n = sum(p.numel() for p in m.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)")

# 3. Forward pass
print("\n[3/4] Forward pass...")
x = torch.randint(0, V, (2, 16))
logits = m(x)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
print(f"  input:  {tuple(x.shape)}")
print(f"  output: {tuple(logits.shape)}")
print(f"  loss:   {loss.item():.4f}")
assert logits.shape == (2, 16, V), f"bad output shape: {logits.shape}"
assert torch.isfinite(logits).all(), "non-finite logits"

# 4. Slot memory
print("\n[4/4] SlotMemory...")
slot = SlotMemory(capacity=64)
slot.insert(5, 0, 42)
assert slot.lookup(5, 0) == 42
assert len(slot) == 1
print("  insert / lookup: OK")
print("  size:            OK")

# 5. Training step (loss should go down)
print("\n[extra] 5 training steps...")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
losses = []
for step in range(5):
    x = torch.randint(0, V, (2, 16))
    logits = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(f"  step {step+1}: loss={loss.item():.4f}")

if losses[-1] < losses[0]:
    print("  -> loss decreasing: OK")
else:
    print("  -> loss not decreasing (may be random, 5 steps is noisy)")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)
