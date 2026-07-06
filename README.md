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

## Inference Engine (C++ / CUDA)

A CUDA C++ inference engine for both custom **GQA** (Llama) and **MLA** (DeepSeek V2) architectures, with full GGUF format support.

```
rina-engine/src/
├── ops/       ← Ops (arch-independent): linear / rms_norm / rope / embedding / silu_mul / saxpy
├── arch/      ← Architecture definitions: GQA (gqa_layer) / MLA (deepseek_mla_layer) / SSM
├── loader/    ← GGUF reader + ArchLoader registry: gqa & mla name mappings
├── infer/     ← Inference orchestrators: GQA engine / MLA engine (Inference interface)
├── core/      ← Infrastructure: buffer / tensor / config / quant
└── main.cpp   ← CLI entry point (uses Inference interface)
```

### Supported Models

| Model | Architecture | Quantization | VRAM |
|-------|-------------|--------------|------|
| Llama 3.2 1B | GQA | fp32 / Q4_0F | 4.7 GB / 1.6 GB |
| DeepSeek-V2-Lite | MLA + MoE | Q2_K / IQ3_XS | 6.4 GB / 7.1 GB |

### Build & Run

```bash
cd rina-engine && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# GQA — Llama 3.2 1B (fp32)
./rina_infer --model /path/to/llama3.2-1b-fp32 \
  --ids "128000 791 7438 315 2324 374" --temp 0.0 --steps 30

# MLA — DeepSeek-V2-Lite (GGUF, Q2_K)
./rina_infer --model DeepSeek-V2-Lite.Q2_K.gguf --gguf \
  --prompt "Hello" --temp 1.0 --topk 40 --steps 15

# GQA — with KV cache quant (q8 / q4 / q2k_q1v)
./rina_infer --model /path/to/llama3.2-1b-fp32 \
  --ids "128000 791 7438 315 2324 374" --kv-quant q8 --steps 30
```

### Quantization Formats

| Format | Type ID | Block Size | BPW | Support |
|--------|---------|------------|-----|---------|
| GGML_Q2_K | 10 | 256 | 2.625 | GPU fused kernel (M=1) + dequant |
| GGML_Q3_K | 11 | 256 | 3.44 | GPU dequant |
| IQ3_XXS | 18 | 256 | 3.06 | GPU dequant |
| IQ3_S | 21 | 256 | 3.44 | GPU dequant |
| GGML_Q4_K | 12 | 256 | 4.5 | GPU dequant |
| GGML_Q5_K | 13 | 256 | 5.5 | GPU dequant |
| GGML_Q6_K | 14 | 256 | 6.6 | GPU dequant |
| IQ4_NL | 20 | 32 | 4.5 | GPU dequant |
| IQ4_XS | 23 | 256 | 4.25 | GPU dequant |
| Q4_0 (RINA) | 5 | 32 | 4.5 | GPU native |
| Q4_0F (RINA) | 7 | 32 | 5.0 | GPU native |

### Build Targets

28 targets total: `rina_infer` + 27 test/alignment targets. All pass.

### Architecture Evolution

| Gen | Architecture | Training | Inference |
|-----|-------------|----------|-----------|
| 1 | SSM (InertiaWave) | PyTorch | — |
| 2 | MoHE (MoE+Attractor) | PyTorch | — |
| 3 | RWKV-v7 | PyTorch | — |
| 4 | AR + Stateful Denoiser | PyTorch | — |
| 5 | MLA + GQA | PyTorch | C++ engine (fp32) |
| 6 | Jamba Hybrid (SSM+Sparse) | PyTorch | C++ engine (q4+q2) |

All training on a single **RTX 3070 Ti Laptop (8 GB VRAM)**.

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
rina/                     ★ PyTorch training (Gen 6 Jamba)
  model_jamba.py          ← Jamba hybrid (SSM+Sparse interleaved)
  model_jamba_qw.py       ← QW extreme quant (q4 weights + LSC q4 + q2+q1 KV)
  model_jamba_qw2.py      ← QW v2 — quant-aware training
  model_jamba_qw3.py      ← QW v3 — multi-stage quant pipeline
  train_jamba.py          ← Recommended Jamba training
  train_jamba_qw.py       ← QW extreme training
  train_jamba_qw2.py      ← QW v2 training
  train_jamba_qw3.py      ← QW v3 training

rina-engine/              ★ C++ inference engine
  src/ops/                ─ Ops: linear / rms_norm / rope / embedding / silu_mul / saxpy
  src/arch/               ─ Arch: GQA (gqa_layer) / MLA (deepseek_mla_layer) / SSM
  src/loader/             ─ Loader: GGUF reader + ArchLoader registry
  src/infer/              ─ Inference: GQA engine / MLA engine (Inference interface)
  src/core/               ─ Core: buffer / tensor / config / quant
  tests/                  ─ 27 test/alignment targets

experiments/               Data pipeline scripts

archive/                   All legacy code (Gen1-Gen5, old experiments, gitignored)

docs/                      Experiment logs & design docs
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

## 推理引擎（C++ / CUDA）

CUDA C++ 推理引擎，支持 **GQA**（Llama）和 **MLA**（DeepSeek V2）两种 Attention 架构，完整 GGUF 格式加载。

```
rina-engine/src/
├── ops/       ← 算子（与架构无关）：linear / rms_norm / rope / embedding / silu_mul / saxpy
├── arch/      ← 架构定义：GQA（gqa_layer）/ MLA（deepseek_mla_layer）/ SSM
├── loader/    ← GGUF 读取 + ArchLoader 注册表：GQA & MLA 名映射
├── infer/     ← 推理编排：GQA 引擎 / MLA 引擎（Inference 接口）
├── core/      ← 基础设施：buffer / tensor / config / quant
└── main.cpp   ← CLI 入口（通过 Inference 接口调用）
```

### 已支持模型

| 模型 | 架构 | 量化 | 显存 |
|------|------|------|------|
| Llama 3.2 1B | GQA | fp32 / Q4_0F | 4.7 GB / 1.6 GB |
| DeepSeek-V2-Lite | MLA + MoE | Q2_K / IQ3_XS | 6.4 GB / 7.1 GB |

### 编译与运行

```bash
cd rina-engine && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# GQA — Llama 3.2 1B (fp32)
./rina_infer --model /path/to/llama3.2-1b-fp32 \
  --ids "128000 791 7438 315 2324 374" --temp 0.0 --steps 30

# MLA — DeepSeek-V2-Lite (GGUF, Q2_K)
./rina_infer --model DeepSeek-V2-Lite.Q2_K.gguf --gguf \
  --prompt "Hello" --temp 1.0 --topk 40 --steps 15

# GQA — 搭配 KV cache 量化 (q8 / q4 / q2k_q1v)
./rina_infer --model /path/to/llama3.2-1b-fp32 \
  --ids "128000 791 7438 315 2324 374" --kv-quant q8 --steps 30
```

### 量化格式支持

| 格式 | 类型 ID | 块大小 | BPW | 支持情况 |
|------|---------|--------|-----|----------|
| GGML_Q2_K | 10 | 256 | 2.625 | GPU 融合 kernel (M=1) + 反量化 |
| GGML_Q3_K | 11 | 256 | 3.44 | GPU 反量化 |
| IQ3_XXS | 18 | 256 | 3.06 | GPU 反量化 |
| IQ3_S | 21 | 256 | 3.44 | GPU 反量化 |
| GGML_Q4_K | 12 | 256 | 4.5 | GPU 反量化 |
| GGML_Q5_K | 13 | 256 | 5.5 | GPU 反量化 |
| GGML_Q6_K | 14 | 256 | 6.6 | GPU 反量化 |
| IQ4_NL | 20 | 32 | 4.5 | GPU 反量化 |
| IQ4_XS | 23 | 256 | 4.25 | GPU 反量化 |
| Q4_0 (RINA) | 5 | 32 | 4.5 | GPU 原生 |
| Q4_0F (RINA) | 7 | 32 | 5.0 | GPU 原生 |

### 构建目标

共 28 个目标：`rina_infer` + 27 个测试/对齐目标，全部通过。

### 架构演进

| 代 | 架构 | 训练 | 推理 |
|----|------|------|------|
| Gen 1 | SSM (InertiaWave) | PyTorch | — |
| Gen 2 | MoHE (MoE+Attractor) | PyTorch | — |
| Gen 3 | RWKV-v7 | PyTorch | — |
| Gen 4 | AR + Stateful Denoiser | PyTorch | — |
| Gen 5 | MLA + GQA | PyTorch | C++ 引擎 (fp32) |
| Gen 6 | Jamba 混合 (SSM+Sparse) | PyTorch | C++ 引擎 (q4+q2) |

全部训练在单张 **RTX 3070 Ti Laptop（8 GB 显存）** 上完成。

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
rina/                     ★ PyTorch 训练（Gen 6 Jamba）
  model_jamba.py          ← Jamba 混合（SSM+Sparse 交错）
  model_jamba_qw.py       ← QW 极压量化（q4 权重 + LSC q4 + q2+q1 KV）
  model_jamba_qw2.py      ← QW v2 — 量化感知训练
  model_jamba_qw3.py      ← QW v3 — 多阶段量化管线
  train_jamba.py          ← 推荐训练脚本
  train_jamba_qw.py       ← QW 极压训练
  train_jamba_qw2.py      ← QW v2 训练
  train_jamba_qw3.py      ← QW v3 训练

rina-engine/              ★ C++ 推理引擎
  src/ops/                ─ 算子：linear / rms_norm / rope / embedding / silu_mul / saxpy
  src/arch/               ─ 架构：GQA（gqa_layer）/ MLA（deepseek_mla_layer）/ SSM
  src/loader/             ─ 加载器：GGUF 读取 + ArchLoader 注册表
  src/infer/              ─ 推理：GQA 引擎 / MLA 引擎（Inference 接口）
  src/core/               ─ 基础设施：buffer / tensor / config / quant
  tests/                  ─ 27 个测试/对齐目标

experiments/               数据管线脚本

archive/                   全部历史代码（Gen1-Gen5、旧实验，gitignored）

docs/                      实验记录与设计文档
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
