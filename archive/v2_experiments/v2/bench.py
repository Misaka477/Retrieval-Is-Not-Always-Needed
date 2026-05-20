"""
V1 vs V2 速度对比 benchmark。

对比:
- V1: RINASeqModel (per-step attractor, M=batch)
- V2: RINASeqModelV2 (batched attractor, M=batch*attract_every)

在固定模型尺寸和 seq_len 下测纯前向 + 反向时间。
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
from model_v2 import RINASeqModelV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DM = 256
NP = 4096
V = 4096
BS = 8
WARMUP = 5
BENCH = 20


def make_toy(seq_len, n=BS):
    return torch.randint(0, V, (n, seq_len))


def measure(name, model, seq_len, do_backward=True):
    x = make_toy(seq_len).to(device)
    model.train()

    for _ in range(WARMUP):
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            x[:, 1:].reshape(-1),
        )
        if do_backward:
            loss.backward()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(BENCH):
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            x[:, 1:].reshape(-1),
        )
        if do_backward:
            loss.backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / BENCH

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:12s} seq={seq_len:4d} | "
          f"{elapsed*1000:7.1f} ms | "
          f"{n_params/1e6:.1f}M params")
    return elapsed


print("=" * 72)
print("Forward + Backward (training)")
print("=" * 72)

for seq_len in [16, 32, 64, 128]:
    print(f"\n--- seq_len={seq_len} ---")

    # V1: standard per-step attractor (attract_every=1 → every step)
    model_v1 = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                            n_slots=V, attract_every=1).to(device)
    measure("V1 ae=1", model_v1, seq_len)

    # V2: adiabatic with attract_every=4
    model_v2_k4 = RINASeqModelV2(V, d_model=DM, n_patterns=NP, beta=0.5,
                                 n_slots=V, attract_every=4).to(device)
    measure("V2 ae=4", model_v2_k4, seq_len)

    # V2: adiabatic with attract_every=8
    model_v2_k8 = RINASeqModelV2(V, d_model=DM, n_patterns=NP, beta=0.5,
                                 n_slots=V, attract_every=8).to(device)
    measure("V2 ae=8", model_v2_k8, seq_len)

    # V2: low-rank r=128 + attract_every=4
    model_v2_lr = RINASeqModelV2(V, d_model=DM, n_patterns=NP, beta=0.5,
                                 n_slots=V, attract_every=4,
                                 pattern_rank=128).to(device)
    measure("V2 ae=4 r128", model_v2_lr, seq_len)


print("\n" + "=" * 72)
print("Forward Only (inference)")
print("=" * 72)

for seq_len in [64, 128, 256, 512]:
    print(f"\n--- seq_len={seq_len} ---")

    model_v1 = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                            n_slots=V, attract_every=1).to(device)
    measure("V1 ae=1", model_v1, seq_len, do_backward=False)

    model_v2_k8 = RINASeqModelV2(V, d_model=DM, n_patterns=NP, beta=0.5,
                                 n_slots=V, attract_every=8).to(device)
    measure("V2 ae=8", model_v2_k8, seq_len, do_backward=False)
