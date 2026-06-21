# RINA — Retrieval Is Not Always Needed

Efficient language modeling with MLA + K→V + GQA + RoPE + SwiGLU + int4 K/Q + int2 V, plus experimental research on tiered memory architectures.

## Research Architecture (Gen 5)

```
                ┌───────────────────────────┐
                │  L3: Latent Indexed       │
                │      Sparse Attention     │  ← A Route ✅ verified
                ├───────────────────────────┤
                │  L2: Full Attention       │
                │      (MLA + K→V + GQA)    │  ← Baseline
                ├───────────────────────────┤
                │  L1: Inertia Wave         │
                │      (SSM recurrence)     │  ← C Route ✅ verified
                └───────────────────────────┘

AC Hybrid (Jamba-style): L1 + L2 + L3 in a single model ✅ verified
```

### Routes

| Route | Description | Status |
|---|---|---|
| **A** | Latent Indexed Attention — contrastive learning on MLA latent (c_kv) for sparse indexing | ✅ v3 triplet margin, gap=0.924 |
| **C** | Inertia Wave — replace attention with decay recurrence, O(T), no KV cache | ✅ trained, quality below attention |
| **AC** | Hybrid combining C (shallow) + Full Attn (mid) + Sparse A (deep) | ✅ verified, inertia not dead |
| **Sparse Inf** | Use A's latent space as index for top-K attention inference | ✅ K=8 ≈ full attn quality, ~40× saving |

### Key Findings

- 1.58-bit ternary quantization collapses at 32M (information channel too narrow)
- int4 K/Q + int2 V works well at 32M
- Latent Indexed Attention: triplet margin > InfoNCE for latent space contrastive learning
- Inertia Wave: 128-dim state insufficient for language modeling alone, but works as auxiliary layers
- AC Hybrid: not better than pure attention at 32M; advantage domain is 7B+ / 100K+ context

## Quick Start

```bash
pip install torch numpy tqdm transformers
```

### Training

```bash
# Baseline (MLA + int4)
python3 -m rina.train --int4 --out models/out-quant --steps 10000 --bsz 4

# Route A (Latent Indexed Attention with triplet contrastive)
python3 -m rina.train_a --int4 --out models/out-rina-a-v3 --steps 10000 --bsz 4

# Route C (Inertia Wave)
python3 -m rina.train_c --out models/out-rina-c --steps 10000 --bsz 4

# AC Hybrid
python3 -m rina.train_ac --int4 --out models/out-rina-ac --steps 10000 --bsz 4
```

### Generation

```bash
python3 -m rina.gen --load models/out-final/rina-gen5-baseline-fp32.pt --int4 --prompt "The capital of France is"
```

### Full Evaluation

```bash
bash eval_all.sh
```

## Project Structure

```
rina/
  model.py       ← Gen 5 baseline (MLA + K→V + quantization)
  model_a.py     ← Route A: Latent Indexed Attention
  model_c.py     ← Route C: Inertia Wave
  model_ac.py    ← AC Hybrid
  train.py       ← baseline training
  train_a.py     ← Route A training (CE + triplet contrastive)
  train_c.py     ← Route C training
  train_ac.py    ← AC Hybrid training
  gen.py         ← generation (works with all routes)

eval_all.sh      ← reproduce generation comparison & ablations

docs/
  RINA实验日志.md ← full experiment log (8800+ lines)

checkpoints/     ← data & weights (gitignored)
models/          ← trained model exports (gitignored)
```

## Hardware

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

---

# RINA — Retrieval Is Not Always Needed

高效语言模型架构 + 层次记忆结构的实验研究。

## 研究架构（Gen 5）

| 路线 | 描述 | 状态 |
|---|---|---|
| **A** | Latent Indexed Attention — 在 MLA latent 空间做对比学习，用于稀疏索引 | ✅ v3 triplet margin 验证通过，gap=0.924 |
| **C** | Inertia Wave — 衰减波递推替代注意力，O(T) 无 KV cache | ✅ 训练通过，质量不及 attention |
| **AC** | Jamba 风格混合（浅层惯性波 + 中层全量注意力 + 深层稀疏索引） | ✅ 验证通过，惯性波非死层 |
| **稀疏推理** | 用 A 的 latent 空间做索引，top-K 稀疏 attention | ✅ K=8 ≈ 全量 attention，~40× 节省 |

### 关键发现

- **1.58-bit 三元量化**在 32M 上不可行（信息通道太窄，权重全部 round 到 0）
- **int4 K/Q + int2 V** 在 32M 上工作正常
- **对比学习策略**：triplet margin > InfoNCE（v3 gap=0.924 vs v1 0.039）
- **惯性波**：128-dim 状态容量不够单独做语言模型，但作辅助层有效
- **AC 混合**：32M 上不优于纯 attention，优势域在 7B+ / 100K+ 长上下文

## 快速开始

```bash
pip install torch numpy tqdm transformers
```

### 训练

```bash
# 基线（MLA + int4）
python3 -m rina.train --int4 --out models/out-quant --steps 10000 --bsz 4

# Route A（Latent Indexed Attention + triplet 对比学习）
python3 -m rina.train_a --int4 --out models/out-rina-a-v3 --steps 10000 --bsz 4

# Route C（惯性波）
python3 -m rina.train_c --out models/out-rina-c --steps 10000 --bsz 4

# AC 混合架构
python3 -m rina.train_ac --int4 --out models/out-rina-ac --steps 10000 --bsz 4
```

### 生成

```bash
python3 -m rina.gen --load models/out-final/rina-gen5-baseline-fp32.pt --int4 --prompt "The capital of France is"
```

### 完整消融对比

```bash
bash eval_all.sh
```

## 包结构

```
rina/
  model.py       ← Gen 5 基线（MLA + K→V + 量化）
  model_a.py     ← Route A: Latent Indexed Attention
  model_c.py     ← Route C: Inertia Wave
  model_ac.py    ← AC 混合架构
  train.py       ← 基线训练
  train_a.py     ← Route A 训练（CE + triplet 对比）
  train_c.py     ← Route C 训练
  train_ac.py    ← AC 混合训练
  gen.py         ← 生成（兼容所有路线）

eval_all.sh      ← 复现生成对比 & 消融实验

docs/
  RINA实验日志.md ← 完整实验记录（8800+ 行）

checkpoints/     ← 数据与权重（gitignored）
models/         ← 训练好的模型导出（gitignored）
```

## 硬件

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## 联系方式

rapidsound@163.com / mikotomisaka477@gmail.com
