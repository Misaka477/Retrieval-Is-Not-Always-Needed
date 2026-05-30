"""
Toy NIAH training — 验证 RINASeqModelV2 的 adiabatic elimination 路径。

对比不同 attract_every 值下的训练速度和 NIAH recall。
"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "test", "v2"))

import torch
import torch.nn.functional as F

from modules.cann_ssm import RINASeqModel
import modules.cann_ssm as _cann_v1
_orig_setup = _cann_v1._setup_cuda_seq_v2
_cann_v1._setup_cuda_seq_v2 = lambda: False
from model_v2 import RINASeqModelV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def make_niah(n, gap, n_keys=10):
    f = 2 * n_keys + 1
    x_list, y_list = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys + 1, (1,)).item()
        v = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()
        x_list.append([k, v] + [f] * gap + [k])
        y_list.append(v)
    return torch.tensor(x_list), torch.tensor(y_list)


def train_one(model_cls, name, attract_every, gaps, model_kwargs=None, is_v1=False):
    if model_kwargs is None:
        model_kwargs = {}
    results = {}
    for gap in gaps:
        t0 = time.time()
        train_x, train_y = make_niah(400, gap)
        test_x, test_y = make_niah(100, gap)

        if is_v1:
            model = model_cls(
                V, d_model=64, n_patterns=1024, beta=0.5,
                n_slots=V_NIAH, attract_every=attract_every, **model_kwargs,
            ).to(device)
        else:
            model = model_cls(
                V, d_model=64, n_patterns=1024, beta=0.5,
                n_slots=V_NIAH, attract_every=attract_every, **model_kwargs,
            ).to(device)
        model.slot_table.zero_()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best = 0.0
        iters = 0
        train_start = time.time()

        for ep in range(60):
            model.train()
            model.zero_grad()
            if is_v1:
                logits = model(train_x.to(device), use_jit=False, use_cuda_seq=False)
            else:
                logits = model(train_x.to(device))
            loss = F.cross_entropy(logits[:, -1], train_y.to(device))
            loss.backward()
            opt.step()
            iters += 1

            with torch.no_grad():
                for b in range(train_x.shape[0]):
                    k = int(train_x[b, 0])
                    v = int(train_y[b])
                    if k > 0 and v > 0:
                        model.slot_write(k, v)

            if ep % 10 == 9:
                model.eval()
                with torch.no_grad():
                    if is_v1:
                        lt = model(test_x.to(device), use_jit=False, use_cuda_seq=False)
                    else:
                        lt = model(test_x.to(device))
                pred = lt[:, -1].argmax(dim=-1)
                acc = (pred == test_y.to(device)).float().mean().item()
                best = max(best, acc)
                elapsed = time.time() - train_start
                it_s = iters / elapsed if elapsed > 0 else 0
                print(f"  [{name}] gap={gap:3d} ep={ep+1:2d} | "
                      f"acc={acc*100:.0f}% best={best*100:.0f}% | "
                      f"{it_s:.1f} it/s")

        elapsed = time.time() - t0
        it_s = iters / elapsed if elapsed > 0 else 0
        results[gap] = {"best": best, "it_s": it_s, "time": elapsed}
        print(f"  [{name}] gap={gap:3d}: best={best*100:.0f}% | "
              f"{it_s:.1f} it/s | {elapsed:.0f}s\n")

    return results


V = 2 * 10 + 2
V_NIAH = V
GAPS = [8, 16, 32]

print("=" * 72)
print("V1 — Baseline (per-step attractor, ae=1)")
print("=" * 72)
r_v1_ae1 = train_one(RINASeqModel, "V1 ae=1", attract_every=1, gaps=GAPS,
                     is_v1=True)

print("=" * 72)
print("V1 — Baseline (per-step attractor, ae=4, 当前 v1 也能间隔)")
print("=" * 72)
r_v1_ae4 = train_one(RINASeqModel, "V1 ae=4", attract_every=4, gaps=GAPS,
                     is_v1=True)

print("=" * 72)
print("V2 — Adiabatic (attract_every=4, gate per-step, attractor batched)")
print("=" * 72)
r4 = train_one(RINASeqModelV2, "V2 ae=4", attract_every=4, gaps=GAPS)

print("=" * 72)
print("V2 — Adiabatic (attract_every=8, 更强批量化)")
print("=" * 72)
r8 = train_one(RINASeqModelV2, "V2 ae=8", attract_every=8, gaps=GAPS)

print("=" * 72)
print("V2 — 低秩 r=128 + attract_every=4")
print("=" * 72)
r_lr = train_one(RINASeqModelV2, "V2 ae=4 r=128", attract_every=4, gaps=GAPS,
                 model_kwargs={"pattern_rank": 128})

print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"{'Gap':>4} | {'V1 ae=1':>10} | {'V1 ae=4':>10} | {'V2 ae=4':>10} | {'V2 ae=8':>10} | {'V2 ae=4 r=128':>14}")
print("-" * 82)
for gap in GAPS:
    v1a1 = r_v1_ae1[gap]["best"] * 100
    v1a4 = r_v1_ae4[gap]["best"] * 100
    a4 = r4[gap]["best"] * 100
    a8 = r8[gap]["best"] * 100
    alr = r_lr[gap]["best"] * 100
    s_v1a1 = r_v1_ae1[gap]["it_s"]
    s_v1a4 = r_v1_ae4[gap]["it_s"]
    s4 = r4[gap]["it_s"]
    s8 = r8[gap]["it_s"]
    slr = r_lr[gap]["it_s"]
    print(f"{gap:4d} | {v1a1:5.0f}% {s_v1a1:4.1f}/s | {v1a4:5.0f}% {s_v1a4:4.1f}/s | "
          f"{a4:5.0f}% {s4:4.1f}/s | {a8:5.0f}% {s8:4.1f}/s | {alr:5.0f}% {slr:4.1f}/s")
