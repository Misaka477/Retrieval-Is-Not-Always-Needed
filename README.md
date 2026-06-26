# RINA — Retrieval Is Not Always Needed

Efficient language modeling with **Jamba-style hybrid**: SSM (InertiaWave) + Sparse MLA Attention + GQA + int4/int2 quantization. Target: all-components-below-4-bit compression — weights, SSM internals, and KV cache.

**Status:** Gen 6 — Jamba hybrid ✅ All base designs verified.

## Research Architecture (Gen 6 — Jamba Hybrid)

```
        12 SSM ×3 ──→ Sparse ×4  (interleaved, no router)
                             │
         ┌── InertiaWave (log-space cumsum, q4)    │  ← 12 layers
         ├── Sparse Gather FA (K=16, W=32, q2+q1)  │  ← 4 layers
         ├── Shared LN + SwiGLU MLP                │
         └── Config: 640d, 16L, 10H, 5KV, 160dc   │
```

### Design Verification

| Design | Status | Evidence |
|--------|--------|----------|
| MLA (d_c 4:1 compression) | ✅ | L2 standalone, CE~5 |
| GQA (2:1 KV ratio) | ✅ | All models native |
| int4(K) + int2(V) | ✅ | L2/L3X/Jamba all pass |
| InertiaWave SSM | ✅ | 12 layers in Jamba, CE 4.8 |
| SSM K=3 train → K=1 inference | ✅ | 1.9× faster, minor quality drop |
| Sparse Gather FA K=16 | ✅ | Standalone + Jamba verified |
| **q2(K)+q1(V) — 3-bit KV cache** | **✅** | **CE 4.2, equals 6-bit KV quality** |
| **LSC q4 SSM (log-space cumsum)** | **✅** | **CE ~5.7, SSM intermediates -75%** |
| Jamba interleaved (SSM×3+Sparse×1) | ✅ | CE 4.8, coherent English |

### Deprecated

| Design | Status | Reason |
|--------|--------|--------|
| CF dynamic routing (conf_head) | ❌ | Training not converged |

## Quick Start

```bash
pip install torch numpy tqdm transformers datasets
```

### Data Preparation

```bash
# Download DCLM (2B tokens, skips existing)
python3 experiments/download_dclm_v2.py 2000

# StarCode (800M)
python3 experiments/download_starcode_v2.py 800

# Math (600M)
python3 experiments/download_math_v2.py 600

# Chinese (600M)
python3 experiments/download_chinese_v2.py 600

# Merge all sources (4B total)
python3 experiments/mix_data_v2.py
```

### Training

```bash
# Jamba baseline (q4+q2 KV)
python3 -u rina/train_jamba.py --steps 50000 --out models/out-rina-jamba-v1

# Jamba v2 — 3-bit KV (q2+q1)
python3 -u rina/train_jamba_v2.py --steps 50000 --out models/out-rina-jamba-v2

# Jamba LSC q4 — log-space SSM + q4 intermediates
python3 -u rina/train_jamba_lq.py --steps 50000 --ssm_q 4 --out models/out-rina-jamba-lq-q4

# Jamba QW extreme — q4 weights + LSC q4 + q2+q1 KV
python3 -u rina/train_jamba_qw.py --steps 50000 --weight_bits 4 --ssm_q 4 --out models/out-rina-jamba-qw
```

### Resume from Checkpoint

```bash
python3 -u rina/train_jamba.py --steps 50000 \
  --resume models/out-rina-jamba-v1/jamba_XXXXX.pt
```

### Generation

Recommended inference parameters: **t=0.3, top_k=20, rep_penalty=5.0**

```python
from rina.model_jamba import RINA_Jamba, RJ_Config
import torch

m = RINA_Jamba(RJ_Config(...)).to('cuda').eval()
sd = torch.load('models/out-rina-jamba-v2/jamba_final.pt')['model']
m.load_state_dict(sd, strict=False)
l, _ = m(x)  # l.shape = [B, T, 128256]
```

## Project Structure

```
rina/
  model_jamba.py       ← ★ Jamba hybrid (SSM+Sparse interleaved)
  model_jamba_lq.py    ← Jamba + log-space SSM + q4 intermediates
  model_jamba_qw.py    ← Jamba + q4 weights + LSC q4 + q2+q1 KV
  model_a.py           ← Route A (full attention MLA)
  model_c.py           ← InertiaWave SSM
  model_l3x.py         ← Sparse Gather FA (K=16, W=32)
  train_jamba.py       ← ★ Recommended Jamba training
  train_jamba_v2.py    ← q2+q1 KV version
  train_jamba_lq.py    ← LSC q4 SSM version
  train_jamba_qw.py    ← QW extreme version
  model_cf.py          ← DEPRECATED: CF routing
  train_cf.py          ← DEPRECATED: CF routing training

experiments/
  download_dclm_v2.py  ← Data download (skip + chunked save)
  download_starcode_v2.py
  download_math_v2.py
  download_chinese_v2.py
  mix_data_v2.py       ← Merge all 4 sources into training npy

docs/
  RINA_实验总览.md      ← Experiment overview (9800+ lines)

checkpoints/           ← weights (gitignored)
models/                ← trained checkpoints (gitignored)
data/                  ← training data (gitignored)
```

## Checkpoints

| File | Size | Description |
|------|------|-------------|
| `out-0.1b-a-v2/a_final.pt` | ~588MB | Route A 800K steps |
| `out-0.1b-c-distil-multistep/c_final.pt` | ~596MB | L1 K=3 distilled |
| `out-rina-jamba-v1/jamba_final.pt` | ~594MB | ★ Jamba baseline, CE 4.8 |
| `out-rina-jamba-v2/jamba_final.pt` | ~594MB | 3-bit KV, CE 4.2 |
| `out-rina-jamba-lq-q4/jambalq_final.pt` | ~594MB | LSC q4 SSM, CE ~5.7 |
| `out-rina-jamba-qw/jambaqw_final.pt` | 🔄 | QW extreme training |

## Hardware

ROG Zephyrus M16 2022 — i9-12900H + RTX 3070 Ti Laptop (8 GB)

## Contact

mikotomisaka477@gmail.com

---

# RINA — Retrieval Is Not Always Needed

高效语言模型架构。**Jamba 式 SSM + 稀疏 Attention 混合**，全套 4-bit 以下量化：权重、SSM 中间量、KV cache。

**当前状态：** Gen 6 — Jamba 混合架构 ✅ 全部底层设计验证通过。

## 研究架构（Gen 6 — Jamba 混合）

```
12 层 SSM ×3 —— 4 层 Sparse ×1 交错（无路由）
                             │
         ┌── InertiaWave（对数空间加法链, q4 中间量化）  ← 12 层
         ├── Sparse Gather FA（K=16, W=32, q2+q1 KV）  ← 4 层
         ├── 共享 LN + SwiGLU MLP                     │
         └── 配置: 640d, 16L, 10H, 5KV, 160dc         │
```

### 量化路线

| 组件 | 当前精度 | 目标 | 压缩比 |
|------|---------|------|--------|
| 权重 | fp32 (32-bit) | **q4 (4-bit)** | 8× |
| SSM 中间量 | fp32 (32-bit) | **LSC q4 (4-bit)** | 8× |
| KV cache | fp32 (32-bit) | **q2+q1 (3-bit)** | 10.7× |
| **总计** | **594MB / 32-bit** | **≤100MB / ~4-bit** | **6-8×** |

### 设计验证

| 设计 | 状态 | 证据 |
|------|------|------|
| MLA（d_c 4:1 压缩） | ✅ | L2 standalone, CE~5 |
| GQA（2:1 KV ratio） | ✅ | 所有模型原生支持 |
| int4(K) + int2(V) | ✅ | L2/L3X/Jamba 全部通过 |
| InertiaWave SSM | ✅ | Jamba 12 层, CE 4.8 |
| SSM K=3 训练 → K=1 推理 | ✅ | 1.9× 加速，质量略降 |
| Sparse Gather FA K=16 | ✅ | 独立测试 + Jamba 验证 |
| **q2(K)+q1(V) 3-bit KV** | **✅** | **CE 4.2, 与 6-bit 持平** |
| **LSC q4 SSM** | **✅** | **CE ~5.7, SSM 中间 -75%** |
| Jamba 混合 (SSM×3+Sparse×1) | ✅ | CE 4.8, 英文通顺 |
| CF 动态路由 | ❌ | 训练未收敛，已弃用 |

## 快速开始

```bash
pip install torch numpy tqdm datasets
```

### 数据准备

```bash
python3 experiments/download_dclm_v2.py 2000    # DCLM 2B tokens
python3 experiments/download_starcode_v2.py 800  # StarCode 800M
python3 experiments/download_math_v2.py 600      # Math 600M
python3 experiments/download_chinese_v2.py 600   # Chinese 600M
python3 experiments/mix_data_v2.py               # 合并为训练数据
```

### 训练

```bash
# Jamba 基线 (q4+q2 KV)
python3 -u rina/train_jamba.py --steps 50000 --out models/out-rina-jamba-v1

# Jamba v2 — 3-bit KV (q2+q1)
python3 -u rina/train_jamba_v2.py --steps 50000 --out models/out-rina-jamba-v2

# Jamba LSC q4 — log-space SSM + q4 中间量化
python3 -u rina/train_jamba_lq.py --steps 50000 --ssm_q 4 --out models/out-rina-jamba-lq-q4

# Jamba QW 极压 — q4 权重 + LSC q4 + q2+q1 KV
python3 -u rina/train_jamba_qw.py --steps 50000 --weight_bits 4 --ssm_q 4 --out models/out-rina-jamba-qw
```

### 从 Checkpoint 续训

```bash
python3 -u rina/train_jamba.py --steps 50000 \
  --resume models/out-rina-jamba-v1/jamba_XXXXX.pt
```

### 生成

推荐推理参数：**t=0.3, top_k=20, rep_penalty=5.0**

```python
from rina.model_jamba import RINA_Jamba, RJ_Config
import torch

m = RINA_Jamba(RJ_Config(...)).to('cuda').eval()
sd = torch.load('models/out-rina-jamba-v2/jamba_final.pt')['model']
m.load_state_dict(sd, strict=False)
l, _ = m(x)  # l.shape = [B, T, 128256]
```

## 包结构

```
rina/
  model_jamba.py       ← ★ Jamba 混合（SSM+Sparse 交错）
  model_jamba_lq.py    ← Jamba + log-space SSM + q4 中间
  model_jamba_qw.py    ← Jamba + q4 权重 + LSC q4 + q2+q1 KV
  model_a.py           ← Route A（全量 attention MLA）
  model_c.py           ← InertiaWave SSM
  model_l3x.py         ← Sparse Gather FA (K=16, W=32)
  train_jamba.py       ← ★ 推荐训练脚本
  train_jamba_v2.py    ← q2+q1 KV 版本
  train_jamba_lq.py    ← LSC q4 SSM 版本
  train_jamba_qw.py    ← QW 极压版本
  model_cf.py          ← 已弃用: CF 路由
  train_cf.py          ← 已弃用: CF 训练

experiments/
  download_dclm_v2.py  ← 下载脚本（跳过已有 + 分块落盘）
  download_starcode_v2.py
  download_math_v2.py
  download_chinese_v2.py
  mix_data_v2.py       ← 合并 4 源为训练 npy

docs/
  RINA_实验总览.md      ← 实验总览（9800+ 行）

checkpoints/           ← 权重（gitignored）
models/                ← 训练好的 checkpoint（gitignored）
data/                  ← 训练数据（gitignored）
```

## Checkpoint 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `out-0.1b-a-v2/a_final.pt` | ~588MB | Route A 800K 步 |
| `out-0.1b-c-distil-multistep/c_final.pt` | ~596MB | L1 K=3 蒸馏 |
| `out-rina-jamba-v1/jamba_final.pt` | ~594MB | ★ Jamba 基线, CE 4.8 |
| `out-rina-jamba-v2/jamba_final.pt` | ~594MB | 3-bit KV, CE 4.2 |
| `out-rina-jamba-lq-q4/jambalq_final.pt` | ~594MB | LSC q4 SSM, CE ~5.7 |
| `out-rina-jamba-qw/jambaqw_final.pt` | 🔄 | QW 极压训练中 |

## 硬件

ROG Zephyrus M16 2022 — i9-12900H + RTX 3070 Ti Laptop (8 GB)

## 联系方式

mikotomisaka477@gmail.com
