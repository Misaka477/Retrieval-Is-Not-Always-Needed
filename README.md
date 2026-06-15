# RINA — Retrieval Is Not Always Needed

Efficient language modeling with MLA + K→V prediction + GQA + RoPE + SwiGLU + 1.58-bit ternary weights + int4 K/Q + int2 V residual.

## Architecture

```
token → Embedding → [12× Block] → LN → Head → logits
Block = LayerNorm → MLA (K→V prediction, GQA, RoPE) → LayerNorm → SwiGLU
```

All linear layers support 1.58-bit ternary quantization (STE). Q/K/V in attention support int4/int2 quantization-aware training.

## Status

| Component | Verified |
|---|---|
| MLA latent compression (d_c=128) | 32M ✅ |
| K→V prediction (no separate V projection) | +0.08 CE vs baseline ✅ |
| GQA (8Q→4KV) | — |
| RoPE | — |
| SwiGLU | — |
| 1.58-bit ternary weights (STE) | replaces all 41 Linear layers ✅ |
| int4 K/Q + int2 V residual | 0 additional CE loss ✅ |

### Baseline vs RINA (90M tokens, GPT-2 vocab, 12L·512D)

| | Standard Transformer | RINA |
|---|---|---|
| Core params | ~38M | **~32M** |
| Total params (w/ emb) | ~63.5M | **~58M** |
| KV cache / token | 1024 | **~192** |
| Weight storage | fp16 | **1.58-bit** |
| Validation CE | 4.17 | **4.25** |

## Quick Start

```bash
pip install torch numpy tqdm transformers
```

```python
from rina import RINA, RINAConfig

cfg = RINAConfig(vocab_size=50257, n_layer=12, n_head=8, n_kv_heads=4, n_embd=512, d_c=128)
model = RINA(cfg)
logits, loss = model(input_ids, targets)

# Enable quantization
cfg.use_158 = True
cfg.use_int4 = True
model = RINA(cfg)
```

## Project Structure

```
rina/
  __init__.py    ← exports RINA, RINAConfig
  model.py       ← model definition
  gen.py         ← generation
  train.py       ← training

docs/
  RINA实验日志.md ← experiment log

checkpoints/     ← data & weights (gitignored)
```

## Hardware

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

---

# RINA — Retrieval Is Not Always Needed

MLA + K→V 预测 + GQA + RoPE + SwiGLU + 1.58-bit 三元权重 + int4 K/Q + int2 V 残差的全栈语言模型。

## 架构

```
token → Embedding → [12× Block] → LN → Head → logits
Block = LayerNorm → MLA (K→V 预测, GQA, RoPE) → LayerNorm → SwiGLU
```

所有权重线性层支持 1.58-bit 三元量化（STE 训练）。Q/K/V 支持 int4/int2 量化感知训练。

## 状态

| 组件 | 状态 |
|---|---|
| MLA 潜压缩（d_c=128） | ✅ 32M 验证 |
| K→V 预测（无独立 V 投影） | ✅ CE +0.08 |
| GQA（8Q→4KV） | ✅ |
| RoPE | ✅ |
| SwiGLU | ✅ |
| 1.58-bit 三元权重（STE） | ✅ 全部 41 个 Linear 替换 |
| int4 K/Q + int2 V 残差 | ✅ 0 CE 额外损失 |

### 基线对比（90M tokens, GPT-2 词表, 12L·512D）

| | 标准 Transformer | RINA |
|---|---|---|
| 核心参数 | ~38M | **~32M** |
| 总参数 | ~63.5M | **~58M** |
| KV cache / token | 1024 | **~192** |
| 权重存储 | fp16 | **1.58-bit** |
| 验证 CE | 4.17 | **4.25** |

## 快速开始

```bash
pip install torch numpy tqdm transformers
```

```python
from rina import RINA, RINAConfig
cfg = RINAConfig(vocab_size=50257, n_layer=12, n_head=8, n_kv_heads=4, n_embd=512, d_c=128)
model = RINA(cfg)
logits, loss = model(input_ids, targets)

# 开启量化
cfg.use_158 = True
cfg.use_int4 = True
model = RINA(cfg)
```

## 包结构

```
rina/        ← 模型代码（MLA + K→V + 量化）
docs/        ← 实验日志
checkpoints/ ← 数据与权重（gitignored）
```

## 硬件

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## 联系方式

rapidsound@163.com / mikotomisaka477@gmail.com
