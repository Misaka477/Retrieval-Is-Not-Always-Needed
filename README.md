# RINA — Retrieval Is Not Always Needed

MoHE (Mixture of Hebbian Experts) — gated dual-memory linear recurrence with Depth-of-Thought.

```python
Fast memory:  h_fast = a·h + b·x                ← SSM gate (per-expert)
Slow memory:  P = patterns.T @ patterns          ← Hebbian linear field
Fusion:       h_out = h_fast + gate·field_force   ← gated dual-memory

Depth-of-Thought: iterative refinement over N passes per token
Winner-take-all Hebbian + loser inhibition → domain specialization
```

## Key Results

### 83M (DM=1024, NP=512, NE=4, GPT-2 50K vocab, weight tying)

| Config | Data | Speed | Status |
|--------|------|-------|--------|
| depth=1 | FineWeb+StarCoder+Math 200M | ~2.95s/step (K3 Light) | ~plateau at loss=8.1, exp_sim collapse |
| **depth=3+noise+dropout** | same | running | **current** |

Expert collapse observed at depth=1 (exp_sim 0.80→0.91, loss flat from step 1000). Mitigated with depth=3, router noise (σ=0.1), expert dropout (p=0.1), aux_loss_weight=0.5.

### 28M (DM=256, NP=512, NE=4, GPT-2 50K vocab)

| Config | Data | ppL | Note |
|--------|------|-----|------|
| depth=1 | WikiText-103 3M | 163.5 | Stable training |
| depth=2 (continuation) | same | **133.3** | +30ppL from depth |
| depth=1 | FW+SC+Math 200M | ~1920 | Pre-K3-Light, ~5.89s/step |

### Legacy: SNN v2 15.3M (CANN + SSM + temporal gating)

| Model | Param | ppl (WikiText-103) | O(T) | Online learning |
|-------|-------|-------------------|------|-----------------|
| SNN v2 | 15.3M | **34.7** | ✅ | ✅ |
| GPT-2 | 14.2M | 34.8 | ❌ | ❌ |

## K3 Light GPU Kernel

K3 Light splits the per-step computation into two stages:

1. **PyTorch batch gates** — all positions × all experts' gate_a/b/proj_in as single batched matmul (cuBLAS)
2. **FusedLightFunction** (attractor only) — `h_fast = a·h + b·x` → field → field_mix → LN → slow_gate → h_out, all in one CUDA kernel

| Version | launches/step | Speed | Backward |
|---------|--------------|-------|----------|
| Old K3 full | 1 (+9 saved tensors) | ~5.89s | manual grad management |
| **K3 Light** | **1 (attractor only)** | **~2.95s (2×)** | **pure autograd** |

Files: `rina/kernels/attractor.py` (forward kernel + backward kernel + FusedLightFunction autograd Function).
Training loop is fully autograd — `finish_training_step()` is a no-op.

### Bug fixes applied to all forward kernels

| Bug | Cause | Fix |
|-----|-------|-----|
| field_mix race | shared memory overwritten by subsequent loop iterations | separate tmp buffer + copy-back |
| LN write race | same pattern | same fix |
| gate bias multiplied 256× | sb[e] added per-thread, then tree-reduction summed all threads | add bias after reduction |

## Training

```bash
# 83M MoHE (current experiment)
python experiments/mohe_83m_run.py     # FineWeb+StarCoder+Math 200M
                                       # depth=3, route_noise=0.1, dropout=0.1

# 28M MoHE (WikiText ablation)
python experiments/mohe_multiexpert.py  # WikiText-103 3M → ppL 133 (depth=2)

# Legacy SNN v2 15.3M
python scripts/train_snn_15m.py        # WikiText-103 38M → ppl 34.7
```

## Package Structure

```
rina/
  mohe.py            — MoHE model (entry point)
  kernels/
    __init__.py       — exports FusedLightFunction
    attractor.py      — K3 Light: fused forward + backward + autograd Function
  sample.py           — adaptive temperature + top-p sampling

rina/ (legacy, deprecated)
  cell.py, model.py, slot.py, config.py, drift.py, niah.py

experiments/
  mohe_83m_run.py     — 83M main training
  mohe_large_run.py   — 28M main training
  mohe_multiexpert.py — WikiText MoHE ablation

archive/dead_rina/    — dead files removed from rina/ (scan.py, cuda_graph.py, data.py, fix_bwd.py)
```

## Hardware

Single **NVIDIA GeForce RTX 3070 Ti Laptop (8 GB VRAM)**.  
83M training: ~2.95s/step, ~670MB steady-state VRAM.

## References

- `docs/RINA实验日志.md` — full experiment log (~6500 lines)
- `docs/RINA_实验总览.md` — condensed experiment overview
- `ORIGIN.md` — project philosophy
- `docs/KVR_实验全记录.md` — predecessor experiment

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

---

# RINA — Retrieval Is Not Always Needed

MoHE（Mixture of Hebbian Experts）——门控双记忆线性递回 + Depth-of-Thought。

```
快记忆: h_fast = a·h_{t-1} + b·x_t          ← SSM 门控（每专家）
慢记忆: P = patterns.T @ patterns             ← Hebbian 联想场
融合:   h_out = h_fast + gate·field_force     ← 门控双记忆

Depth-of-Thought: 每 token 多轮迭代精化
赢家通吃 Hebbian + 输家抑制 → 领域自然分化
```

## 实验结果

### 83M（DM=1024, NP=512, NE=4, GPT-2 50K 词表, weight tying）

| 配置 | 数据 | 速度 | 状态 |
|------|------|------|------|
| depth=1 | FineWeb+StarCoder+Math 200M | ~2.95s/step（K3 Light） | loss=8.1 平台期，专家趋同 |
| **depth=3+noise+dropout** | 同上 | 运行中 | **当前** |

depth=1 时专家趋同（exp_sim 0.80→0.91，step 1000 起 loss 持平）。
已通过 depth=3、router noise（σ=0.1）、expert dropout（p=0.1）、aux_loss_weight=0.5 缓解。

### 28M（DM=256, NP=512, NE=4, GPT-2 50K 词表）

| 配置 | 数据 | ppL | 备注 |
|------|------|-----|------|
| depth=1 | WikiText-103 3M | 163.5 | 稳定训练 |
| depth=2（续训） | 同上 | **133.3** | 二轮迭代 +30 ppL |
| depth=1 | FW+SC+Math 200M | ~1920 | K3 Light 前，~5.89s/step |

### 前代：SNN v2 15.3M（CANN + SSM + 时序门控）

| 模型 | 参数量 | ppl（WikiText-103） | O(T) | 在线学习 |
|------|--------|-------------------|------|---------|
| SNN v2 | 15.3M | **34.7** | ✅ | ✅ |
| GPT-2 | 14.2M | 34.8 | ❌ | ❌ |

## K3 Light GPU 算子

K3 Light 将每步计算拆为两阶段：

1. **PyTorch batch gates** — 所有位置×所有 expert 的 gate_a/b/proj_in 一次 batched matmul（cuBLAS）
2. **FusedLightFunction**（attractor only）— `h_fast = a·h + b·x` → field → field_mix → LN → slow_gate → h_out，一个 CUDA kernel 完成

| 版本 | launch 数 | 速度 | 反向 |
|------|----------|------|------|
| 旧 K3 完整 | 1（+9 中间 tensor） | ~5.89s | 手动 grad 管理 |
| **K3 Light** | **1（attractor only）** | **~2.95s（2×）** | **纯 autograd** |

文件：`rina/kernels/attractor.py`（前向 kernel + 反向 kernel + FusedLightFunction autograd Function）。
训练循环全 autograd — `finish_training_step()` 是 no-op。

### 修复的所有前向 kernel bug

| Bug | 原因 | 修复 |
|-----|------|------|
| field_mix 竞争 | 共享内存被后续循环覆写 | 独立 tmp buffer + 拷回 |
| LN 写竞争 | 同上 | 同上 |
| gate bias 被 256 个线程重复加 | sb[e] 每个线程初始，tree reduction 累加 256× | 改为 reduction 后加 bias |

## 训练

```bash
# 83M MoHE（当前实验）
python experiments/mohe_83m_run.py     # FineWeb+StarCoder+Math 200M
                                       # depth=3, route_noise=0.1, dropout=0.1

# 28M MoHE（WikiText 消融）
python experiments/mohe_multiexpert.py  # WikiText-103 3M → ppL 133（depth=2）

# 前代 SNN v2 15.3M
python scripts/train_snn_15m.py        # WikiText-103 38M → ppl 34.7
```

## 包结构

```
rina/
  mohe.py            — MoHE 模型（入口）
  kernels/
    __init__.py       — 导出 FusedLightFunction
    attractor.py      — K3 Light：融合前向+反向+autograd Function
  sample.py           — 自适应温度 + top-p 采样

rina/（前代，已标注 deprecated）
  cell.py, model.py, slot.py, config.py, drift.py, niah.py

experiments/
  mohe_83m_run.py     — 83M 主线训练
  mohe_large_run.py   — 28M 主线训练
  mohe_multiexpert.py — WikiText MoHE 消融

archive/dead_rina/    — 从 rina/ 移出的死文件（scan.py, cuda_graph.py, data.py, fix_bwd.py）
```

## 硬件

单张 **NVIDIA GeForce RTX 3070 Ti Laptop（8 GB VRAM）**。  
83M 训练：~2.95s/step，~670MB 稳态显存。

## 参考

- `docs/RINA实验日志.md` — 完整实验记录（~6500 行）
- `docs/RINA_实验总览.md` — 整理版实验概览
- `ORIGIN.md` — 项目哲学
- `docs/KVR_实验全记录.md` — 前代实验

## 联系方式

rapidsound@163.com / mikotomisaka477@gmail.com
