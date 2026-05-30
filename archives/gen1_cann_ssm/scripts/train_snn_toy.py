"""
SNN-CANN Toy NIAH 验证 — 脉冲门控 vs baseline.

测量:
  1. Recall vs gap (NIAH)
  2. 脉冲率 (spike_rate) 训练中变化
  3. 速度对比
"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from modules.snn_cell import SNNSeqModel
from modules.cann_ssm import RINASeqModel
import modules.cann_ssm as _c; _c._setup_cuda_seq_v2 = lambda: False

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


def train_model(model, name, gap, steps=80, mini_bs=32):
    train_x, train_y = make_niah(400, gap)
    test_x, test_y = make_niah(100, gap)
    model.to(device)
    model.slot_table.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0.0
    t0 = time.time()

    for ep in range(steps):
        model.train()
        model.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), mini_bs):
            idx = perm[i:i + mini_bs]
            logits = model(train_x[idx].to(device))
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()

        with torch.no_grad():
            for b in range(train_x.shape[0]):
                k, v = int(train_x[b, 0]), int(train_y[b])
                if k > 0 and v > 0:
                    model.slot_write(k, v)

        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                lt = model(test_x.to(device))
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best = max(best, acc)
            sr = model.get_spike_rate() if hasattr(model, 'get_spike_rate') else 1.0
            print(f"  [{name}] gap={gap:3d} ep={ep+1:2d}: acc={acc*100:.0f}% best={best*100:.0f}% spike={sr*100:.0f}%")

    elapsed = time.time() - t0
    return best, elapsed


V = 2 * 10 + 2  # 22

for gap in [8, 16, 32]:
    print(f"\n── gap={gap} ──")

    # Baseline (standard CANN)
    model_baseline = RINASeqModel(V, d_model=64, n_patterns=1024, beta=0.5,
                                  n_slots=V, attract_every=1).to(device)
    best_base, t_base = train_model(model_baseline, "BASELINE", gap)
    print(f"  BASELINE gap={gap}: best={best_base*100:.0f}% ({t_base:.0f}s)")

    # SNN
    model_snn = SNNSeqModel(V, d_model=64, n_patterns=1024, beta=0.5,
                            n_slots=V, attract_every=1).to(device)
    best_snn, t_snn = train_model(model_snn, "SNN", gap)
    print(f"  SNN      gap={gap}: best={best_snn*100:.0f}% ({t_snn:.0f}s)")

    print(f"  Δ recall: {best_snn - best_base:+.0%}  speed: {t_base/max(t_snn, 1):.1f}x")
