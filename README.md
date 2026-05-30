# RINA (Retrieval Is Not Always Needed) — MoHE-RWKV

109M language model with per-token expert routing, depth-of-thought iteration, and RWKV-v7 linear recurrence.

## Architecture

```
Embed → WKV (RWKV-v7) → [depth×3: Router → 12×AttractorExpert → Consolidate] → Head
```

| Component | Detail |
|-----------|--------|
| Time mixing | RWKV-v7 WKV CUDA kernel (head 64, H=12, float32) |
| Expert | Attractor: `h + gate · field`, `field = h @ (P^T·P) → Proj → FieldMix → LN` |
| Routing | Per-token, topk=2, ×3.0 scaling, no cross-step smoothing |
| Depth | 3 iterations with `h = h_new` chain (DoT-like) |
| Params | 109.44M (embed 50.3M + expert 55.0M + others 4.1M) |

## Current Results

### Gen 3 — MoHE-RWKV 109M (current)

- **ppl=4.9**, val_ppl=4.3, route entropy=0.5 (differentiated)
- **SEQ=512, BSZ=4, ~2 it/s** (54× throughput over previous arch)
- 500M tokens: FW(50%) + SC(20%) + Math(15%) + Chinese(15%)
- LR: 3e-4 → cosine → 3e-5 over 90000 steps

### Gen 2 — MoHE 28M/91M (archived)

4-expert MoE with GPT-2 vocab. Plateau at ppl≈3665.

### Gen 1 — CANN-SSM 15M (archived)

Hybrid SSM + CANN with temporal gating. ppl=34.7 on WikiText-103.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (resume from checkpoint or from init)
python experiments/mohe_transferred_train.py

# Generate
python experiments/generate.py

# Prepare data (500M FW+SC+Math+Chinese)
python experiments/prepare_data_rwkv.py

# Weight transfer (RWKV-7 → MoHE-RWKV init)
python experiments/weight_transfer.py
```

## Package

```
rina/
  __init__.py            ← from rina import MoHERWKV, sample, TRIE_TOKENIZER
  model.py               ← MoHE-RWKV (WKV7Fn + AttractorExpert + MoHERWKV)
  sample.py              ← Adaptive temperature + top-p sampling
  rwkv_tokenizer.py      ← RWKV trie tokenizer (rwkv_vocab_v20230424)

experiments/             ← Current generation scripts
  mohe_transferred_train.py
  generate.py
  prepare_data_rwkv.py
  weight_transfer.py

kernels/                 ← WKV CUDA kernel (float32, head=64)
  rwkv7_clampw.cu / .cpp

archives/                ← Previous generations
  gen1_cann_ssm/         (CANN-SSM 15M-25M)
  gen2_mohe/             (MoHE 28M/91M, GPT-2 vocab)
```

## Hardware

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## References

- `docs/RINA实验日志.md` — full experiment log
- `docs/RINA_实验总览.md` — condensed experiment overview
- `docs/ARCH.md` — architecture & training recipe
- `ORIGIN.md` — project philosophy

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

---

# RINA (Retrieval Is Not Always Needed) — MoHE-RWKV

109M 语言模型，逐 token 专家路由、深度迭代精化、RWKV-v7 线性递回。

## 架构

```
Embed → WKV (RWKV-v7) → [depth×3: Router → 12×AttractorExpert → Consolidate] → Head
```

| 组件 | 细节 |
|------|------|
| 时间递回 | RWKV-v7 WKV CUDA kernel (head 64, H=12, float32) |
| 专家 | Attractor: `h + gate · field`, `field = h @ (P^T·P) → Proj → FieldMix → LN` |
| 路由 | 逐 token, topk=2, ×3.0 缩放, 无跨步平滑 |
| 深度迭代 | 3 轮 `h = h_new` 链式传递 (DoT-like) |
| 参数量 | 109.44M (embed 50.3M + expert 55.0M + 其他 4.1M) |

## 当前结果

### Gen 3 — MoHE-RWKV 109M (当前)

- **ppl=4.9**, val_ppl=4.3, 路由熵=0.5 (已分化)
- **SEQ=512, BSZ=4, ~2 it/s** (前代 54× 吞吐)
- 500M token: FW(50%) + SC(20%) + Math(15%) + 中文(15%)
- LR: 3e-4 → cosine → 3e-5, 90000 步

### Gen 2 — MoHE 28M/91M (已归档)

4 专家 MoE + GPT-2 词表。平台期 ppl≈3665。

### Gen 1 — CANN-SSM 15M (已归档)

SSM + CANN 混合 + 时序门控。WikiText-103 ppl=34.7。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练 (自动续训或从初始化权重开始)
python experiments/mohe_transferred_train.py

# 生成测试
python experiments/generate.py

# 准备数据 (FW+SC+Math+中文 500M)
python experiments/prepare_data_rwkv.py

# 权重迁移 (RWKV-7 → MoHE-RWKV)
python experiments/weight_transfer.py
```

## 包结构

```
rina/
  __init__.py            ← from rina import MoHERWKV, sample, TRIE_TOKENIZER
  model.py               ← MoHE-RWKV (WKV7Fn + AttractorExpert + MoHERWKV)
  sample.py              ← 自适应温度 + top-p 采样
  rwkv_tokenizer.py      ← RWKV trie tokenizer (rwkv_vocab_v20230424)

experiments/             ← 当前代实验脚本
  mohe_transferred_train.py
  generate.py
  prepare_data_rwkv.py
  weight_transfer.py

kernels/                 ← WKV CUDA kernel (float32, head=64)
  rwkv7_clampw.cu / .cpp

archives/                ← 前代归档
  gen1_cann_ssm/         (CANN-SSM 15M-25M)
  gen2_mohe/             (MoHE 28M/91M, GPT-2 词表)
```

## 硬件

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## 参考

- `docs/RINA实验日志.md` — 完整实验记录
- `docs/RINA_实验总览.md` — 整理版实验概览
- `docs/ARCH.md` — 架构与训练配方
- `ORIGIN.md` — 项目哲学

## 联系方式

rapidsound@163.com / mikotomisaka477@gmail.com
