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

## Key Results (15.3M)

| Model | ppl | O(T) inference | Content-addressable memory | Online learning |
|-------|-----|---------------|---------------------------|-----------------|
| RINA (SNN v2) | **34.7** | ✅ | ✅ (slot) | ✅ (Hebbian) |
| V1 CANN-SSM | 34.5 | ✅ | ✅ (slot) | ❌ |
| GPT-2 | 34.8 | ❌ O(T²) | ❌ | ❌ |
| SSM-only (ablation) | 34.7 | ✅ | ⚠️ | ❌ |

**Memory cost:** RINA slot is independent of sequence length. Transformer 70B with 1M context requires KV cache ≈ **2.6 TB**; RINA slot requires only **16 GB**, regardless of context length.

**Slot limitation (honest):** The current slot mechanism does not autonomously decide what to store or retrieve. It requires manual `slot_write()` calls and only injects at the last position. It cannot independently track conversation context. This is a known limitation — autonomous content-addressable memory remains future work.

### Cross-distribution results

| Task | RINA 15M | GPT-2 15M | Improvement |
|:-----|:---------|:----------|:------------|
| WikiText-103 ppl | 33.3 | 34.8 | +5% |
| Code zero-shot (StarCoder) | **65.80** | **14,432** | **219×** |
| seq=512 inference ppl | **36.0** | 104.0 | −65% |
| Scalability (FineWeb 137M tokens) | 57.8→43.45 | plateaued at ~280M | new scaling law |
| Code generation capacity | 5.03 ppl (trained) | — | — |

> **Evaluation verified:** TinyLLaMA 1.1B on WikiText-103 via the same pipeline gives ppl=8.0 (seq=1024), within published range. All evaluation numbers are measured consistently across models.

## Quick Start

Pretrained checkpoints: [github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases](https://github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases)

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

## Seq-Len Benchmark

```bash
python scripts/bench_seqlen.py       # sequence-length benchmark
```

## V1 Baselines

> V1 (CANN-SSM) is RINA's predecessor — SSM gate + attractor + slot without temporal gating or Hebbian plasticity. Kept for direct ablation and comparison.

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
  train_fineweb.py          — FineWeb scaling law experiment
  train_code.py             — StarCoder code training
  train_code_seq128.py      — progressive seq=128 code training
  train_code_seq256.py      — progressive seq=256 code training
  train_cann_15m.py         — V1 baseline
  train_ablation.py         — SSM-only ablation
  train_gpt2_15m.py         — GPT-2 baseline
  bench_seqlen.py           — sequence-length benchmark
  bench_niah_*.py           — NIAH benchmarks (V1 and SNN v2)
  bench_code_ppl.py         — cross-distribution code ppl comparison (--seq, --th)
  bench_ppl_fineweb.py      — FineWeb distribution validation
  bench_wikitext_ppl.py     — cross-checkpoint WikiText comparison
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

## v3 MoHE (Active Development)

MoHE (Mixture of Hebbian Experts) is the current architecture — gated dual-memory linear recurrence:

```
Fast memory:  h_fast = a·h_{t-1} + b·x_t          ← SSM gate
Slow memory:  P = patterns.T @ patterns             ← Hebbian field
Fusion:       h = h_fast + gate·(h_fast @ P)        ← gated dual-memory

Depth-of-Thought: iterative refinement over N passes
MoHE: 4 experts, winner-take-all Hebbian + loser inhibition
```

**Key results** (28M, GPT-2 50K vocab, depth=1→2):
- WikiText-103 (3M tokens): **ppL 133.3** (ep12, stable training)
- FineWeb+StarCoder+OpenWebMath (200M, running): **ppL ~1920** (ep1/5)

**Key innovations:** winner-take-all Hebbian (domain specialization), Depth-of-Thought (hidden-space iteration, not token-space CoT), linear field `h @ patterns.T @ patterns` (no softmax → associative-scan friendly), expert inertia (routing smoothing), online Hebbian (model learns after deployment).

**Experiments:** `experiments/mohe_large_run.py` (200M main), `experiments/mohe_multiexpert.py` (WikiText MoHE), `experiments/selfplay_dual_memory.py` (linear field proof, ppL 93.4).

### K3 GPU Kernel Optimization

The per-expert forward computation (K1+K2) was fused into a single **K3 kernel** — 1 CUDA launch vs 8 per step:

| Version | launches/step | Speed-up |
|---------|--------------|----------|
| Python baseline | ~1280 | 1× |
| K1+K2 | ~640 | ~1.1× |
| **K3 forward** | **1** | **~2×** |

**Training**: hybrid approach — fused CUDA forward + Python backward via `FusedExpertFunction` (torch.autograd.Function). Gradient precision < 1e-6.

**K4 (head batching):** head projection moved out of position loop, batched into single GEMM (M=512 vs M=8). Inference: **4.2 it/s** (+68%).

**Files:** `rina/kernels/` (K3 forward), `rina/kernels/train.py` (training Function), `rina/mohe.py` (integrated). K1+K2 removed (K3 `ne=1` replaces both).

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

**记忆显存：** RINA slot 不随序列长度增长。Transformer 70B 在 1M 上下文时 KV cache ≈ **2.6 TB**；RINA slot 仅 **16 GB**，与上下文长度无关。

**Slot 局限性（诚实说明）：** 当前 slot 不会自主判断该存什么、该查什么。需要外部手动调用 `slot_write()`，且仅在最后一位自动注入。无法独立追踪对话上下文。自主内容寻址记忆是未解决的问题。

### 跨分布结果

| 评测 | RINA 15M | GPT-2 15M | 差距 |
|:-----|:---------|:----------|:------|
| 代码零样本迁移 (StarCoder) | **65.80** | **14,432** | **219×** |
| seq=512 推理 ppl | **36.0** | 104.0 | −65% |
| FineWeb 137M tokens 缩放 | 57.8→43.45 | ~280M 平台 | 缩放不收敛 |
| 代码生成上限 | 5.03 ppl (已训) | — | — |

## 快速开始

预训练权重：[github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases](https://github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases)

```
reproduce.bat                    # 一键：装依赖 + 冒烟测试
python scripts/train.py          # 训练（13 epoch, ~10h）
python scripts/warm_restart.py   # 续训拿最佳 ppl 34.7
python scripts/generate.py       # 生成 demo

> 注意：15.3M 参数量（RINA 和 GPT-2 都是）无法产出流畅长文本。15M 下生成质量不是有意义的指标，ppl 才是可靠的语言建模评估标准。

### V1 基线

> V1 (CANN-SSM) 是 RINA 的前代架构 — SSM gate + attractor + slot，不含 temporal 门控和 Hebbian 可塑性。保留用于消融和直接对比。

```
python scripts/train_cann_15m.py     # V1 CANN-SSM → ppl 34.5
python scripts/train_ablation.py     # SSM-only  → ppl 34.7
python scripts/train_gpt2_15m.py     # GPT-2     → ppl 34.8
python scripts/bench_seqlen.py       # 序列长度 benchmark
```
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

## v3 MoHE（当前开发中）

MoHE（Mixture of Hebbian Experts）——门控双记忆线性递回，当前主线架构：

```
快记忆: h_fast = a·h + b·x          ← SSM 门控
慢记忆: P = patterns.T @ patterns   ← Hebbian 联想场
融合:   h = h_fast + gate·(h @ P)   ← 门控双记忆

Depth-of-Thought: 隐藏空间迭代精化
MoHE: 4 专家，赢家通吃 Hebbian + 输家抑制
```

**实验结果（28M, GPT-2 50K 词表）：**
- WikiText-103（3M tokens）：**ppL 133.3**（ep12，稳定训练）
- FineWeb+StarCoder+OpenWebMath（200M，正在跑）：**ppL ~1920**（ep1/5）

**实验：** 
- `experiments/mohe_large_run.py` — 200M tokens 主线
- `experiments/mohe_multiexpert.py` — WikiText MoHE（ppL=133）
- `experiments/selfplay_dual_memory.py` — 单层线性场验证（ppL=93.4）

### K3 GPU 算子优化

将每步 4 专家的前向计算（原 K1+K2 × 4 = 8 次 launch）融合为 **1 次 launch**：

| 版本 | launches/step | 加速比 |
|------|--------------|--------|
| Python baseline | ~1280 | 1× |
| K1+K2 | ~640 | ~1.1× |
| **K3 forward** | **1** | **~2×** |

训练使用混合方案：fused CUDA forward + Python backward（`FusedExpertFunction` autograd Function），梯度精度 < 1e-6。

**文件：** `rina/kernels/`（K3 forward）、`rina/kernels/train.py`（训练 autograd Function）、`rina/mohe.py`（集成）。K1+K2 已删除。K4 head batch 将推理提升至 **4.2 it/s**。

## 参考

- `docs/RINA实验日志.md` — 完整实验记录（4274 行）
- `ORIGIN.md` — 项目哲学与技术动机
- `docs/KVR_实验全记录.md` — 前代实验

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

