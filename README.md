# RINA — Retrieval Is Not Always Needed

CANN + SSM + temporal SNN gating + Hebbian plasticity.  
A recurrent architecture with contraction-guaranteed attractor dynamics.  
15.3M parameters, **ppl 34.7** on WikiText-103 (equals GPT-2 15M).

## Architecture

```
ż = −ε · A(z)                   ← conceptual unification
                                 (actual computation below)

ε = ‖h̃ − h‖ / ‖h‖               ← relative prediction error
h̃ = a·h + b·(x·W_p)            ← SSM gate
A(h̃) = h̃ + α·(S(h̃·Pᵀ)·P − h̃)  ← attractor
δP = η·ε·(h̃ − P_k) − ½η·ε·repel  ← Hebbian + inhibition
```

Five core components:
1. **SSM gate** — cross-dim mixing, token-level transform
2. **CANN attractor** — global contraction, basin retrieval via softmax @ P
3. **Temporal SNN** — ε-gated adaptive sparsity (att=10-26%)
4. **Hebbian plasticity** — online learning, winner pattern pulled toward state
5. **Lateral inhibition** — repels neighbors to prevent pattern collapse

## Key Results (15.3M on WikiText-103)

| Model | ppl | O(T) inference | Content-addressable memory | Online learning |
|-------|-----|---------------|---------------------------|-----------------|
| RINA (SNN v2) | **34.7** | ✅ | ✅ (slot) | ✅ (Hebbian) |
| V1 CANN-SSM | 34.5 | ✅ | ✅ (slot) | ❌ |
| GPT-2 | 34.8 | ❌ O(T²) | ❌ | ❌ |
| SSM-only (ablation) | 34.7 | ✅ | ⚠️ | ❌ |

Multi-key NIAH (gap=128, random positions): **RINA 100%** vs GPT-2 36% (−47%).

## Quick Start

Pretrained checkpoints: [github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases/tag/v0.1.0](https://github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases/tag/v0.1.0)

```bash
git clone ...
cd RINA_Project

# Option A: one-click environment check
reproduce.bat

# Option B: manual
pip install -r requirements.txt
python scripts/quick_test.py     # 10s smoke test (no external data)
```

## Training

```bash
# Full training (13 epoch, ~10h on RTX 3070 Ti)
python scripts/train.py          # → ppl ~35.4 at ep10

# Warm-restart for best result (34.7 ppl)
python scripts/warm_restart.py   # → ppl 34.7 at ep12
```

Training produces `checkpoints/cann_snn15m_v2_final.pt`.

## Generation

> Note: 15.3M models (RINA and GPT-2 alike) are too small to produce fluent long text.
> Generation quality is not a meaningful metric at this scale — ppl is the reliable metric.
> See Section 5 of the training log for generation examples at 15M.

```bash
python scripts/generate.py       # requires trained checkpoint
```

## NIAH Benchmark

```bash
# Toy NIAH: quick verification
python scripts/bench_niah_snn_slot.py

# Real-text + extreme + multi-key NIAH
python scripts/bench_niah_snn_final.py
```

## V1 Baselines

```bash
python scripts/train_cann_15m.py     # V1 CANN-SSM → ppl 34.5
python scripts/train_ablation.py     # SSM-only  → ppl 34.7
python scripts/train_gpt2_15m.py     # GPT-2     → ppl 34.8
python scripts/bench_seqlen.py       # seq-len benchmark
```

## ⚠️ Critical: Import Order

Torch **must** be imported after `tokenizers` and `datasets`:

```python
os.environ["HF_DATASETS_OFFLINE"] = "1"
from tokenizers import Tokenizer       # before torch
from datasets import load_dataset
import torch                           # after datasets
torch.manual_seed(42)
```

CUDA+multiprocessing fork will deadlock silently if this order is violated.

## Package Structure

```
rina/                        — clean API package
  __init__.py               — exports TemporalSNNCell, TemporalSNNModel, SlotMemory
  cell.py                   — TemporalSNNCell (ε-gated attractor + Hebbian)
  model.py                  — TemporalSNNModel (train + generate + slot)
  slot.py                   — SlotMemory (dict fallback backend)
  data.py                   — WikiText-103 loading + BPE tokenizer
  config.py                 — JSON config loader

modules/                    — V1 reference + backward compat
  cann_ssm.py               — V1 CANN-SSM (RINASeqModel, CUDA wrappers)
  temporal_snn_cell.py      — re-exports from rina/ (old scripts still work)

scripts/
  train.py                  — main training (15.3M temporal SNN v2)
  train_snn_15m.py          — same as train.py, imports from modules/ (alias)
  warm_restart.py           — LR-reset continuation (34.5 → 34.7)
  generate.py               — autoregressive generation demo
  quick_test.py             — 10s smoke test (no external data)
  train_cann_15m.py         — V1 baseline
  train_ablation.py         — SSM-only ablation
  train_gpt2_15m.py         — GPT-2 baseline
  bench_seqlen.py           — sequence-length benchmark
  bench_niah_slot.py        — V1 NIAH toy
  bench_niah_realtext.py    — V1 NIAH real-text
  bench_niah_extreme.py     — V1 NIAH random position
  bench_niah_multikey.py    — V1 NIAH multi-key
  bench_niah_snn_slot.py    — SNN v2 NIAH toy
  bench_niah_snn_realtext.py — SNN v2 NIAH real-text
  bench_niah_snn_final.py   — SNN v2 NIAH extreme + multi-key
  train_multimodal.py       — multi-modal proof-of-concept

config/
  default.json              — verified optimal hyperparameters

archive/                    — abandoned experiments (not deleted)
  modules/                  — dead code paths (snn_cell, rina_v3, etc.)
  scripts/                  — debug/legacy scripts
  cuda/                     — CUDA kernel source + build artifacts
  v2_experiments/           — adiabatic/DMD/linear K experiments
  ablation/                 — optimization ablation studies
  deq/                      — DEQ implicit differentiation experiments
  checkpoints/              — legacy model weights

reproduce.bat               — one-click: pip install → quick_test → info
requirements.txt            — torch, tokenizers, datasets, tqdm, transformers
```

## Hardware

Developed and tested on a single **NVIDIA GeForce RTX 3070 Ti Laptop (8 GB VRAM)**.  
Training time: ~10 hours for 13 epochs on WikiText-103 (38M tokens).

## References

- `docs/RINA实验日志.md` — full experiment log (May 15-21, 2026, 4274 lines)
- `ORIGIN.md` — project philosophy and technical motivation
- `docs/KVR_实验全记录.md` — predecessor experiment (KVR)

---

# RINA — Retrieval Is Not Always Needed

CANN + SSM + temporal SNN 门控 + Hebbian 可塑性。  
15.3M 参数，WikiText-103 上 **ppl 34.7**（持平 GPT-2 15M）。

## 架构

```
ż = −ε · A(z)                    ← conceptual unification
                                   (actual computation below)

ε  = ‖h̃ − h‖ / ‖h‖               ← relative prediction error
h̃  = a·h + b·(x·W_p)            ← SSM gate
A(h̃) = h̃ + α·(S(h̃·Pᵀ)·P − h̃)   ← attractor
δP = η·ε·(h̃ − P_k) − ½η·ε·repel ← Hebbian + inhibition
```

五个核心组件：SSM 门控（交叉维度混合） → CANN 吸引子（全局收缩 + basin 检索） → Temporal SNN（ε 门控自适应稀疏） → Hebbian 可塑性（在线学习） → 侧抑制（防 pattern collapse）。

## 实验结果

| 模型 | ppl | O(T) 推理 | 内容寻址记忆 | 在线学习 |
|------|-----|----------|------------|---------|
| RINA (SNN v2) | **34.7** | ✅ | ✅ (slot) | ✅ (Hebbian) |
| V1 CANN-SSM | 34.5 | ✅ | ✅ (slot) | ❌ |
| GPT-2 15M | 34.8 | ❌ O(T²) | ❌ | ❌ |
| 消融 (SSM-only) | 34.7 | ✅ | ⚠️ | ❌ |

多 key NIAH（gap=128, 随机位置）：**RINA 100%** vs GPT-2 36%（−47%）。

## 快速开始

预训练权重：[github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases/tag/v0.1.0](https://github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases/tag/v0.1.0)

```
reproduce.bat                    # 一键：装依赖 + 冒烟测试
python scripts/train.py          # 训练（13 epoch, ~10h）
python scripts/warm_restart.py   # 续训拿最佳 ppl 34.7
python scripts/generate.py       # 生成 demo

> 注意：15.3M 参数量（RINA 和 GPT-2 都是）无法产出流畅长文本。15M 下生成质量不是有意义的指标，ppl 才是可靠的语言建模评估标准。
```

## ⚠️ import 顺序（防静默死锁）

`tokenizers` 和 `datasets` **必须**在 `torch` 之前导入。

## 包结构

```
rina/           — 新 API 包（cell, model, slot, data, config）
modules/        — V1 参考 + 向后兼容（cann_ssm.py, temporal_snn_cell.py 从 rina/ re-export）
scripts/        — 训练/评测/基线脚本
config/         — default.json（已验证最优超参）
archive/        — 废弃实验，未删除
```

## 硬件

单张 **NVIDIA GeForce RTX 3070 Ti Laptop (8 GB VRAM)** 开发。  
WikiText-103（38M tokens）训练约 10 小时。

## 参考

- `docs/RINA实验日志.md` — 完整实验记录（4274 行）
- `ORIGIN.md` — 项目哲学与技术动机
- `docs/KVR_实验全记录.md` — 前代实验

