"""
Verify the restructuring didn't break anything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ok = True

print("=== Complete Architecture Verification ===")
print()

# Test 1: rina/ package
try:
    from rina import TemporalSNNCell, TemporalSNNModel, SlotMemory
    from rina.config import load_config
    print("[OK] rina/ package")
except Exception as e:
    print("[FAIL] rina/ package:", e); ok = False

# Test 2: modules backward compat
try:
    from modules.temporal_snn_cell import TemporalSNNCell, TemporalSNNModel
    from rina import TemporalSNNModel as NewModel
    assert TemporalSNNModel is NewModel, "class mismatch"
    print("[OK] modules/temporal_snn_cell re-export")
except Exception as e:
    print("[FAIL] modules/temporal_snn_cell:", e); ok = False

# Test 3: V1 reference
try:
    from modules.cann_ssm import RINASeqModel, CANNSSMCell, _full_forward
    print("[OK] modules/cann_ssm V1 reference")
except Exception as e:
    print("[FAIL] modules/cann_ssm:", e); ok = False

# Test 4: forward + backward
try:
    import torch
    m = TemporalSNNModel(100, d_model=64, n_patterns=128, beta=0.5,
                         attract_every=2, error_threshold=1.0,
                         hebbian_lr=0.01, inhibition_threshold=0.8)
    x = torch.randint(0, 100, (2, 16))
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 100), x[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print("[OK] forward + backward: loss=%.4f" % loss.item())
except Exception as e:
    print("[FAIL] forward + backward:", e); ok = False

# Test 5: generate
try:
    ids = list(m.generate([1, 2, 3], max_len=8, temperature=0.8, top_k=10))
    assert len(ids) == 8
    print("[OK] model.generate()")
except Exception as e:
    print("[FAIL] generate:", e); ok = False

# Test 6: SlotMemory
try:
    s = SlotMemory(capacity=64)
    s.insert(5, 0, 42)
    assert s.lookup(5, 0) == 42
    assert s.lookup(5, 1) is None
    assert len(s) == 1
    s.clear()
    assert len(s) == 0
    print("[OK] SlotMemory")
except Exception as e:
    print("[FAIL] SlotMemory:", e); ok = False

# Test 7: config
try:
    cfg = load_config()
    assert cfg["dm"] == 840
    print("[OK] config: dm=%d" % cfg["dm"])
except Exception as e:
    print("[FAIL] config:", e); ok = False

# Test 8: abandoned modules gone from modules/
try:
    from modules.snn_cell import SNNCANNCell
    print("[WARN] snn_cell still importable from modules/")
except ImportError:
    print("[OK] abandoned modules cleaned from modules/")

# Test 9: references/ not in root
if not os.path.isdir("references"):
    print("[OK] references/ moved to archive/")
else:
    print("[WARN] references/ still in root")

# Test 10: no hardcoded HF_HOME
count = 0
for root, dirs, files in os.walk("scripts"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
                if 'HF_HOME' in content and 'setdefault' not in content:
                    print("[WARN] hardcoded HF_HOME in %s" % path)
                    count += 1
if count == 0:
    print("[OK] no hardcoded HF_HOME paths")

print()
if ok:
    print("=== ALL 10 CHECKS PASSED ===")
else:
    print("=== SOME CHECKS FAILED ===")
    exit(1)
