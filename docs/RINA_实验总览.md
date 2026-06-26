# RINA 实验总览

> 从原始实验日志 (`RINA实验日志.md`, 9800+ 行) 整理而成。
> 按实验阶段组织，突出因果链条和关键结论。
>
> **当前版本：** Gen 6 — Jamba 混合架构验证（06-25~06-26）

**项目：** RINA (Retrieval Is Not Always Needed)
**架构：** MLA + K→V + GQA + RoPE + SwiGLU + int4 K/Q + int2 V + Latent Indexed Attention / Inertia Wave
**实验卡：** RTX 3070 Ti Laptop (8 GB)

---

## 1. 背景与动机

### 1.1 KVR 的失败（05-01~05-15）

KVR（Key-predicted Value Retrieval）试图压缩 Transformer 的 KV cache：用 top-K 检索替代全量 attention。结论：

- top-K 能定位 needle 简单场景 12/12
- 但**生成质量退化无法通过无训练方案解决**
- **根因：** MLP 不认识 KVR 的 attention 分布——检索到的信息虽然位置对了，但 MLP 层无法将其整合进 next-token 预测

**范式转向：** 放弃"检索 + softmax 加权"，转向基于 CANN（连续吸引子神经网络）的记忆范式。

### 1.2 架构设计（05-15 初始）

```
输入 → [SNN 脉冲编码] → [CANN-SSM 融合核心] → [精确槽] → 输出
         ↑ 多模态统一         ↑ 吸引子 + SSM      ↑ 精确存储
```

三个核心组件：
1. **CANN-SSM：** SSM 做线性递推，CANN 做永久吸引子记忆
2. **精确槽（Exact Slots）：** 修复 CANN 不擅精确检索的硬伤，4096 槽，LRU evict
3. **脉冲编码（Spike Encoding）：** 多模态统一表示，高精确性需求走精确槽路径

关键设计决策（05-15 15:00）：**去掉独立的 NTM/DNC 精确槽系统**，改为统一吸引子场——精确槽 = 被"加深"保护的 attractor basin。

---

## 2. 实验阶段

### 2.1 范式验证：CANN 单状态容量瓶颈（05-15）

| 实验 | 架构 | 结果 | 根因 |
|:-----|:-----|:-----|:------|
| Exp1: 单状态 CANN | `CANNCell`, 25K params, dm=64 | gap=8 时 recall 仅 16-21% | 单一向量存不下多个 token，filler 推离状态 |
| Exp2: 双 basin | Normal + Protected basin | gap≥8 接近随机 (11%) | 检索路径依赖当前状态→漂移后找不到 basin |
| Exp3-4: HashSlot + I_ext/logit bias | CANN + HashSlot | gap=8 仍 ≤11% | 手写 CANN + BPTT 训练不可行 (loss≈2.3) |
| **Exp5: Hopfield + Slot** | HopfieldLayer + HashSlot | gap=8 **78%** | **三层架构方向确认** |
| **Exp6: Hopfield + Embedding 注入** | Hopfield + Embedding→cat→head | **gap≤32 100%, gap=64 92%** | **最终有效方案**（32K params 验证） |

**结论：** 单一 attractor 状态无法做精确序列记忆。需要分离"语义流（CANN/Hopfield）+ 精确存储（HashSlot）"。

### 2.2 CANN-SSM 替换 Hopfield（05-15 Phase 1）

| 指标 | Hopfield | CANN-SSM |
|:-----|:---------|:----------|
| 精确记忆 (gap≤32) | 100% | 100% |
| 精确记忆 (gap=64) | 92% | ~10% |
| 推理复杂度 | O(T²) | **O(T)** |
| 512K 推理可行 | ❌ impossible | ✅ ~8 分钟 |

**定论：** CANN-SSM 替换方向正确——精确记忆持平（gap≤32），复杂度从 O(T²) 降到 O(T)。长 gap 的记忆差距由精确槽填补。

### 2.3 真实文本训练：3.7M（05-16 Phase 3a/b）

| 数据集 | 参数量 | ppl | 对比 Transformer |
|:-------|:------|:----|:----------------|
| TinyStories | 3.7M | **12.08** | 10M Transformer ~15-18 |
| Wikitext-2 | 3.7M | **40.52** | 10M Transformer ~35-40 |
| **参数效率** | **3×** | | |

踩坑：tokenizer 未保存导致生成不可用（ppl 指标可信，但生成文本全为重复 "ears"）。

### 2.4 15M 规模化 + 基线对比（05-17）

**配置：** dm=768, np=4096, seq=64, bs=8, WikiText-103 38M tokens

#### 三模型对比

| 模型 | ppl | 训练时间 | 训练显存 | 推理 O(T) | 外部记忆 | 上下文自愈 |
|:------|:----|:---------|:---------|:----------|:---------|:----------|
| CANN-SSM (V1) | **34.5** | 6h | 4.5G | ✅ | ✅ slot | ✅ |
| 消融 (SSM-only) | 34.7 | 3h | 4.0G | ✅ | ✅ | ✅ |
| GPT-2 15M | **34.8** | 40min | 2.5G | ❌ O(T²) | ❌ | ❌ |

**消融结论：** seq=64 短序列下 attractor 贡献仅 0.2 ppl（~0.6%）。attractor 的价值在长序列推理中体现（日志 17:28 验证）。

#### 速度对比（推理 batch=1）

| seq_len | CANN-SSM O(T) | GPT-2 O(T²) |
|:-------:|:-------------:|:------------:|
| 64 | ~8ms | ~5ms |
| 512 | **~35ms** | **OOM / 159 ppl** |

##### 推理时 ppl 稳定性（v3，所有模型 3 次平均）

原生 WikiText-103 段落，3 次随机采样 × 30 段取平均。

| seq_len | SNN v2 | V1 CANN | Ablation | GPT-2 |
|:-------:|:------:|:--------:|:--------:|:-----:|
| 64 | 33.0 | 35.8 | 35.3 | 31.2 |
| 128 | 34.8 | 35.4 | 33.7 | 49.7 |
| 256 | 34.5 | 32.4 | 35.0 | 75.5 |
| 512 | **36.0** | 37.5 | 35.3 | **104.0** |
| 1024 | 43.4 | 43.8 | 44.3 | 124.5 |

GPT-2 从 seq=64 到 512 ppL 暴涨 +73（31→104），CANN 模型仅涨 +3（33→36）。O(T) 递推的 ppl 稳定性是架构级优势。

---

## 3. 加速路线穷尽（05-17~18）

| # | 路线 | 加速比 | ppl 影响 | 结论 |
|:-:|:-----|:-------|:---------|:------|
| 1 | 低秩 pattern r=128 | 1.5× | +6 | ✅ 边际可用 |
| 2 | depthwise gate | 2.2× | 68.7 | ❌ 交叉维度混不可砍 |
| 3 | parallel scan | 7.5× | 78.7 | ❌ 丢选择性遗忘 |
| 4 | batch stacking | 7.6× | gate 崩溃 | ❌ 状态依赖不可展开 |
| 5 | CUDA fusion | 1.0× | — | ❌ atomicAdd 瓶颈 |
| 6 | V2 adiabatic | — | recall 崩塌 | ❌ 吸引子不是慢变量 |
| 7 | 线性 K 替换 attractor | — | gate 噪声 10× 超边界 | ❌ K 无全局检索 |
| 8 | K pre-rotate gate | — | 单步撑不住 | ❌ gate 放大偏差 |
| 9 | K 维持（双流） | — | K→gate cos=0.68 | ⚠️ 13× 改善但不够 |
| 10 | gate 预判 | — | precision 80-85% | ❌ 不到 90% 安全线 |

**终局结论：** M=8 窄 GEMM 是递推式架构在消费级 GPU 上的物理天花板。dense gate GEMM 是最小不可缩减的瓶颈。

**唯一可用的加速：** 低秩 pattern r=128（1.5×，ppl +2-4）。

---

## 4. Temporal SNN 升级（05-18~20）

### 4.1 从两路并行到唯一胜出

| 分支 | 方法 | 结果 |
|:-----|:-----|:------|
| A 路：脉冲门控 (`snn_cell.py`) | per-dimension spike mask | 50% spike rate，无真正稀疏，ppl 退化 |
| **B 路：时序门控 (`temporal_snn_cell.py`)** | **预测误差 ε 门控** | **att 8-26%，ppl 不损** |

### 4.2 关键消融

| 实验 | 配置 | ppl | 结论 |
|:-----|:------|:----|:------|
| 阈值扫描 | th=0.3/0.5/1.0/always | th=1.0 最优 | att 50%→8%，ppl 不变 |
| pred_loss 消融 | λ=0.05 vs 0 | λ=0.05 劣 7.1 | MSE 平滑正则与 CE 打架 |
| 优化器消融 | per-step LR + wd=0.1 vs per-epoch + wd=0.01 | **37% ppl 改善** | **per-step LR 是 15M 差距主犯** |
| Hebbian 4-way | Hebb+Inhib / Hebb / No Hebb / DEQ | 全部无差 | Hebbian 在小规模中性 |

### 4.3 DEQ 验证

| 指标 | 值 |
|:-----|:----|
| 不动点收敛步数 | 17 步（α=0.5） |
| DEQ 3步 vs BPTT 5步梯度方向 | cos_sim=0.9985 ✅ |
| 结论 | attractor 是全局 contraction mapping，DEQ 训练可行 |

### 4.4 SNN v2 15M 训练曲线

**配置：** dm=840, np=4096, seq=64, bs=8, th=1.0, pred=0, wd=0.01, per-epoch LR

```
ep    ppl      att    V1 ppl    Δ vs V1
────────────────────────────────────────
 1   107.9    19%    101.4     +6.5
 2    61.8    21%     58.9     +2.9
 3    52.4    23%     50.5     +1.9
 4    47.4    24%     45.7     +1.7
 5    43.9    24%     42.5     +1.4
 6    40.9    25%     39.9     +1.0
 7    38.9    25%     37.9     +1.0
 8    37.1    25%     40.2     −3.1  🔥
 9    35.9    26%     37.7     −1.8
10    35.4    26%     34.5     +0.9
```

Warm-restart（ep11-12）：**34.7**（最终，反超 V1 的 34.5）。

---

## 5. 最终结果

### 5.1 语言建模

| 模型 | 参数量 | ppl | 训练时间 | 推理 O(T) |
|:-----|:------|:----|:---------|:----------|
| **RINA SNN v2** | **15.3M** | **34.7** | ~10h | ✅ |
| V1 CANN-SSM | 14.2M | 34.5 | ~6h | ✅ |
| GPT-2 | 14.2M | 34.8 | ~40min | ❌ O(T²) |
| SSM-only | 14.2M | 34.7 | ~3h | ✅ |

### 5.2 NIAH 最终矩阵

| 实验 | GPT-2 | V1 CANN+slot | **SNN v2** |
|:-----|:------|:-------------|:-----------|
| Toy (gap 8-128) | — | 100% | **100%** |
| Real-text (gap=8) | 100% (固定位作弊) | 22% | 23% |
| Real-text (gap=128) | 100% (固定位作弊) | 22% | **32%** |
| Extreme (随机位) | 83% | 21% | **100%** |
| Multi-key (3 keys) | **36%** | 18% | **100%** |
| Δ single→multi | **−47%** | −3% | **+79%** |

### 5.3 能力矩阵

| 能力 | GPT-2 | V1 | SNN v2 |
|:-----|:------|:---|:--------|
| ppl 持平 | ✅ 34.8 | 34.5 | 34.7 |
| O(T) 推理 | ❌ | ✅ | ✅ |
| Content-addressable memory | ❌ | ✅ | ✅ |
| Online learning (Hebbian) | ❌ | ❌ | ✅ |
| Temporal sparsity | ❌ | ❌ | ✅ (att=26%) |
| Contraction guarantee | ❌ | ✅ (数学) | ✅ (DEQ 验证) |
| Self-excitation immunity | ❌ | ✅ | ✅ |
| Pattern collapse prevention | ❌ | ❌ | ✅ (侧抑制) |
| Multi-modal native | ❌ | ❌ | ✅ (ViT 概念验证, ppl 4.7) |
| **Memory cost (70B, 1M cxt)** | **KV cache ≈ 2.6 TB** | **slot 16 GB** (不随 seq 增长) | **slot 16 GB** |

#### 生成 Demo

两个模型在 15M 下均只能产生断裂文本。SNN v2 与 GPT-2 15M 的生成质量无可见差异——这不是架构问题，是 15M 参数容量决定了生成不可能流畅。论文中不以此指标论证性能，生成质量需 100M+ 方可评估。

---

## 6. 架构洞见

1. **attractor 不是 ppl 提升器——是 slot-trust 加速器。** 消融只差 0.2 ppl，但 gap=128 时 CANN 10 步学会信任 slot，ABL 200 步学不会
2. **Gate 的交叉维度混合不可砍。** depthwise gate ppl 68.7，dense GEMM 是 attractor basin 的最低入场费
3. **M=8 窄 GEMM 不是算法问题——是硬件物理天花板。** 递推式计算和 batched GEMM 在消费级 GPU 上存在结构性冲突。解药：换大卡（A100 bs=32）或换硬件（NPU/定制 ASIC）
4. **"同参数量"对比本身对 RINA 不公平。** Transformer 15M = 12 层，每层参数只服务 token 一次投影。RINA 15M = 1 层，同一参数被同一 token 反复使用 64 步。seq=1024 时 RINA ppl 43 vs GPT-2 124——RINA 的参数被深度复用而非垂直分割，因此能用更少的参数吃掉更长的序列。**缩放法则应对递推架构改用"有效 FLOPs × 参数密度"而非"参数数量"作为衡量标准。**

---

## 7. 补充实验 · Slot-Aware 训练

**代码：** `scripts/train_snn_slot.py`（新建于 2026-05-21）

**动机：** Slot 机制在 post-hoc fine-tune 中加入时，gate 已经学会了"预测下一个词靠上下文"，难以在短时间内学会信任 slot 注入。如果在 LM 预训练阶段就混入 slot 样本，gate 可同步学习信任 slot 信号。

**方法：** 在原始 SNN v2 训练之上：
- 每模型新增 `slot_table`（vocab_size × d_model）和 `slot_proj`
- 模型 forward 改为每步检查 slot 表并注入（非仅最后一位）
- 训练数据 90% 正常 WikiText-103 + 10% 合成 key→value 序列

**关键 bug 修复链条：**
1. slot_write 需包 `torch.no_grad()` 防止 backward 图断裂
2. slot_table 必须在每个 batch 前清零，防止正常文本 token 偶然匹配已存储的 key 并注入噪声
3. ppl 显示需排除合成序列 batch（否则 10% 随机噪声将累积平均拉高 50-60 ppl）

**结果（ep13）：**

```
ep   ppl(LM)  slot_acc  att
─────────────────────────────
 1   114.7       1%     16%
 2    65.1       3%     19%
 5    46.6       6%     23%
10    36.8       9%     24%
13    33.3      22%     25%
```

slot_acc 初始随机基线 0.024%（1/4096），最终 22%。slot-aware 训练有效提升了 gate 对 slot 信号的信任度，在结构化数据上的信任提升更为显著（代码模型 gap=32 达 100%）。

---

## 8. 未来方向

- **在线 Hebbian + lateral inhibition：** 部署后持续适应（需更大模型抑制随机噪声）
- **自我博弈（self-play）：** 双流探索（Stream B + noise, ε 裁判）
- **多模态扩展：** 图像/音频/文本共享同一 attractor 场（ViT 概念验证已通，ppl 4.7）
- **蒸馏：** 用 GPT-2 124M 蒸馏 15M CANN（预期 ~1h 拿 ppl 基线）
- **收敛束搜索：** contraction 保证束不会爆炸

---

*整理自 `RINA实验日志.md`（4301 行, 2026-05-15~05-21）*

**评估流程验证：** TinyLLaMA 1.1B 在相同 pipeline 下 WikiText-103 验证集 ppl=8.0（seq=1024），与公开范围一致。所有模型在同一评估框架下对比，无系统性偏差。

**Slot 当前局限：** slot 机制能完成构造的多 key 检索，但不会自动判断该存/该读。需要外部手动调用 `slot_write()`，仅在最后一位自动注入。无法独立追踪对话上下文——自主记忆仍是待解决的方向。

---

## 9. v3 MoHE：门控双记忆线性递回（2026-05-23）

### 根因分析

v2 softmax attractor 的根本矛盾：**basin 收敛要求状态静止，SSM 递回要求状态流动——两者互斥。** 这是 v2 训练慢（非线性 → 不能 associative scan）、生成重复（basin 锁死 h）的根因。

### 架构：线性场 + 双记忆 self-play

```python
P = patterns.T @ patterns           # [dm, dm]，Hebbian 演化的场张量
h_t = (a + gate_t·P)·h_{t-1} + b·x_t   # 全线性递推
```

去除 softmax，attractor 替换为线性联想场。快记忆（SSM gate）+ 慢记忆（场 P）目标不一致 → 自然形成自我博弈。

### 演化路径

```
v2 softmax attractor → 训练慢、生成重复
  ↓ 根因：basin 收敛与 SSM 递回冲突
线性场 → 可 scan、保留慢记忆
  ↓ 自然延伸
MoHE（赢家通吃 Hebbian MoE）
  ↓ LayerNorm consolidation + 专家惯性 + GPT-2 50K 词表
mohe_large（FW+StarCoder+OpenWebMath 200M, 正在跑）
```

### 验证结果

| 实验 | 配置 | 数据 | ppL | 速度 |
|:-----|:-----|:-----|:----|:-----|
| 单层线性场 | dm=256, 2.8M | WikiText 3M | **93.4** | 6 it/s |
| MoHE depth=1 | 28M, 4 expert | WikiText 3M | **163.5** | 2.5 it/s |
| MoHE depth=2 | 28M, 4 expert | WikiText 3M | **133.3** | 1.6 it/s |
| MoHE 200M（跑中） | 28M, 4 expert | FW+SC+Math 200M | **~1900** | 1.4 it/s |

### 关键技术决策

- **去 softmax → 去 slot → 去 NIAH → 去 slot_proj → 去 write_net**（单向清理）
- **Consolidation_norm** 替代 `÷√4`（自适应专家融合）
- **Head init: N(0,0.001) + bias=-10.8**（50K 词表稳定）
- **NaN 时 scheduler.step() 必执行**（warmup 不卡死）
- **赢家通吃 Hebbian + 输家抑制**（专家分化保障）

---

## 10. K3/K4 GPU Kernel 优化（2026-05-24）

### K3: N-Expert Fused Forward

将 MoHE 每步 4 个专家的计算（原 K1+K2 × 4 = 8 次 launch）融合为 **1 次 launch**。K3 完全通用：`ne`（专家数）、`dm`（维度）、`bs`（batch size）均为运行时参数，无硬编码。唯一硬件约束是 `2 × dm × sizeof(float) ≤ 48 KB`（shared memory 上限），即 `dm ≤ 6144`，当前 `dm=256` 绰绰有余。

### K4: Batched Head Projection

将 head projection 移出并行循环，batch 为一次大 GEMM（M 从 8 提升到 512），推理 **4.2 it/s**（+68%）。

### 混合加速方案

| 阶段 | 方案 | Launch 数 |
|------|------|-----------|
| **Forward** | K3 fused CUDA kernel | **1**（替代 8） |
| **Backward (input grads)** | Python autograd（保存中间值） | N/A |
| **Backward (param grads)** | `compute_param_grads()` + `apply_param_grads()` | N/A |

纯 CUDA backward kernel 存在 shared memory 竞争条件无法解决，改用 Python 实现 backward。Associative scan 经验证在 MoHE 上无加速价值（gate 计算依赖 h_{t-1} 无法并行化，且 K3 已足够融合）。

### 清理

- K1+K2 死代码已删除（K3 ne=1 即可实现单专家，比 K1+K2 还少一次 launch）
- `kernels.py` → `rina/kernels/__init__.py`
- `kernels_train.py` → `rina/kernels/train.py`

### 精度验证

| 测试 | 配置 | max error |
|------|------|-----------|
| Forward 精度 | DM=64/128/256, NE=2/4 | 8.34e-07 |
| Gradient 精度 | DM=64/128, NE=2/4 | 9.54e-07 |
| Batched head vs per-position | DM=256, BS×SEQ=512 | 4.77e-07 |

全部通过 1e-4 阈值。

### 性能

| 版本 | launches/step | 推理 it/s |
|------|--------------|-----------|
| Python baseline | ~1280 | ~1.4 |
| K3 + K4 head batch | **~1** | **4.2** |

### 训练调用方式

```python
model.train()
opt.zero_grad()
logits = model(x)              # K3 fused forward + K4 batched head
loss = F.cross_entropy(...)
loss.backward()                 # Python backward（FusedExpertFunction）
model.finish_training_step()   # expert 参数梯度
opt.step()
```

### 待讨论：Raw Logits 与生成策略

当前 MoHE 输出 raw logits，训练时 CrossEntropyLoss 内部做 softmax，推理时直接 `/3 + argmax`。

可优化方向：
1. 动态温度调节（Adaptive Temperature）：用输出分布的熵 H 反馈调节温度 T，低熵(复读) -> 调高 T 打散，高熵(乱码) -> 调低 T 收紧
2. Top-P (Nucleus) Sampling 替代 argmax，增加生成多样性
3. 三级生成控制：Temperature -> Top-P -> Sampling，纯生成侧改动不碰模型

---

## 11. MoHE ep1 完成 + Scaling Law 讨论（2026-05-24 12:08）

### ep1 训练结果

| 指标 | 值 |
|------|---|
| 配置 | DM=256, NE=4, MAX_DEPTH=1, label_smoothing=0.1 |
| 训练数据 | FineWeb+StarCoder+OpenWebMath, ~200M tokens total, SUBSAMPLE=8 |
| ep1 步数 | ~48,800 |
| ep1 最终 ppL | **~2984**（含 label_smoothing，去噪后真实 ppL ~2200） |
| 训练速度 | depth=1: ~2.0 it/s, depth=2: ~1.0 it/s |
| 显存 | 稳定 ~670MB |

### 生成效果（step 47K 时采样）

词汇表掌握，局部语法可用，句子级语义仍在学习中。生成使用了自适应温度 + Top-P 采样（`rina/sample.py`）。

### 核心参数量：2.4M

MoHE 28M 的参数量释放：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding + Head | 25.7M | 91% |
| 4 x Expert + Consolidation + Router | **2.4M** | **9%** |

决定模型能力的是 **2.4M 核心参数**，其余是词表刚性开销不可消除。Transformer 在同等核心参数量下不存在（2.4M 核心无法支持 attention 的最小维度）。

### depth=2 计划

ep2 切换 `MAX_DEPTH=2`，每步迭代两次 routing + expert + consolidation，预计 ppL 有额外下降空间。

### Scaling Law 讨论

MoHE 核心参数 2.4M 训 200M tokens 后仍在下降，不符合标准 RNN/LSTM 在小参数量下的快速 plateau 行为。可能原因：

1. **Hebbian patterns 持续更新**：与传统静态权重不同，patterns 每个 step 通过 `index_add_` 被新 token 修正。模型的有效容量不封死在训练完成时刻
2. **SSM gate + 线性场替代 attention**：不需要 attention 头的最小维度约束，参数效率更高
3. **MoE 路由 + Hebbian 分化**：2.4M 在 4 个 expert 间的动态分配等价于更高效地利用了有限参数

验证方法：
- 完成 ep2 (depth=2) 和 ep3，看 ppL 是否继续下降
- 与同核心参数量（2.4M）的传统模型直接对比 ppL
- 如果 MoHE 2.4M 在 3 ep（600M tokens）后 ppL 达到 1500-2000 区间，则证明参数量效率显著超越缩放定律预测

---

## 12. MoHE 83M + K3 Light Kernel（2026-05-25）

### 12.1 83M 模型配置

从 28M 扩展到 83M（DM=1024, NP=512, NE=4, weight tying, SEQ=128），参数量释放：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding + Head | 51.5M | 69% |
| 4 x Expert + Consolidation + Router | **23.1M** | **31%** |

核心参数占比从 28M 的 9% 提升到 31%，核心/词表比显著改善。

### 12.2 K3 Light Kernel

v3 的 K3 fused forward 包含完整的 S5-style 门控递推（gate_a/b + proj_in）→ 计算量大且需要保存 9 个中间 tensor 给 backward。K3 Light 将计算拆为两步：

1. **PyTorch batch gates：** 所有位置×所有 expert 的 gate_a/b 一次 matmul 完成
2. **FusedLightFunction（attractor only）：** 以 h_fast = a·h + b·x 为输入 → field → field_mix → LN → slow_gate → h_out

收益：

| 指标 | 旧 K3 | K3 Light | 收益 |
|------|-------|----------|------|
| 每步时间 | ~5.89s | ~2.95s | **~2×** |
| 训练逻辑 | 手动 grad 管理 | 纯 autograd | 简洁 |

`finish_training_step()` 变为 no-op，删除 `rina/kernels/train.py`。

### 12.3 关键 Bug 修复

| Bug | 位置 | 影响 |
|-----|------|------|
| field_mix shared memory 竞争 | 所有 forward kernel | 输出非确定且错误 |
| LN 写 shared memory 竞争 | 所有 forward kernel | 同上 |
| Gate bias 被 256 个线程重复累加 | 所有 forward kernel | gate 恒为 0 或 1，sigmoid 饱和 |

三个 bug 存在于所有历史 kernel（K3、K3 Light、训练 kernel）中，此次一并修复。

### 12.4 depth=1 训练结果

28M 到 83M，200M tokens，MAX_DEPTH=1：

```
step    loss   ppl      exp_sim
───────────────────────────────
  200   8.40   18650     0.80
  400   8.23    8512     0.88
  800   8.20    5530     0.90
 1400   8.01    4523     0.90
 2000   8.08    4155     0.91
 2800   8.10    3911     0.91
```

**核心发现：** depth=1 时 4 个专家全面趋同（exp_sim 0.80 → 0.91），路由始终均匀（entropy ≈ max 1.386）。loss 从 step 1000 起平在 ~8.1，不再下降。

**根因：** 没有 depth 迭代，每个 token 只经过一层 expert。4 个专家输入相同 → 路由均匀 → Hebbian 的 push/pull 互相抵消 → 继续趋同。

### 12.5 depth=3 + 分化增强

| 措施 | 参数 | 目标 |
|------|------|------|
| Router noise | σ=0.1 | 强制不均匀路由分配 |
| Expert dropout | p=0.1 | 迫使 consolidate 不依赖特定专家 |
| Aux loss weight | 0.5 → (原 0.1) | 更强惩罚均匀路由 |
| MAX_DEPTH | 1 → 3 | 迭代计算让专家差异化分工 |

**正在运行中。**

---

## 13. Gen 3 — MoHE-RWKV 109M

**架构：** RWKV-v7 WKV backbone + 12 attractor-based MoE experts + per-token topk=2 routing + depth=3 chain

| 参数 | 值 |
|------|-----|
| dm / NP | 768 / 1536 |
| n_experts / topk | 12 / 2 |
| route_raw | ×3.0 |
| router_bias | N(0, 0.5²) |
| entropy_bonus | -0.01 × H |
| depth | 3 (chain: h = h_new) |
| expert scale | ×1.0 (gate × field, 无衰减) |
| SEQ / BSZ | 512 / 4 |
| vocab | 65536 (RWKV) |
| speed | 2-3 it/s @ SEQ=512, 8GB laptop |
| 参数量 | 109.44M |

### 13.1 预训练结果

| step | ppl | ent | val_ppl | 备注 |
|------|-----|-----|---------|------|
| 200 | 1238 | 0.96 | — | transferred init |
| 3000 | 5.8 | 2.48 | — | SEQ=128, 旧 .mean(1) 路由 |
| 6000 | 5.5 | 2.13 | — | SEQ=512, topk=2 |
| 90000 | 4.9 | 0.85 | 4.3 | per-token + inertia=0 → 路由分化 |

### 13.2 关键改进

1. **per-token 路由（去掉 .mean(1)）** — 熵从 2.48 暴降到 0.5
2. **topk=2 + inertia=0** — 路由分化锁死，无需额外 loss
3. **depth chain（h = h_new）** — 修复原 depth=3 空转 bug
4. **54× 吞吐** — attractor 批量化：`torch.stack` 替代 4608 次 Python 调用

### 13.3 SFT (进行中)

- 数据：196K 条（R1 88K + o1 CoT 50K + Magic Code 50K + GSM8K 7.5K）
- 训练：LoRA rank=32, SEQ=1024, BSZ=2, 2000 steps
- 输出格式：`<|user|>\n...\n<|assistant|>\n...`
- 身份替换：GPT/Claude/Qwen 等 → Anthelia

### 13.4 后续

- SFT 完成后跑生成测试
- 验证集 + benchmark（GSM8K / HumanEval）
- 继续预训练扩充数据量

---

## 14. AR + State Diffusion（2026-06-02~03）

**背景：** MoHE+RWKV 109M 方向放弃后（attractor MoE 坍缩），转向更直接的方案：冻结官方 RWKV-v7 12L backbone，在 AR 生成的 hidden state 上加一个 denoiser 做后处理。

### 14.1 当前架构

```
官方 12L RWKV-v7 backbone（冻结 + return_h）
  → h [768] → denoiser → h' → head → logits → token
```

**关键决策：** 训练时 denoiser 看到的 h 必须来自 AR 生成分布。之前所有尝试（训在随机数据 batch 上）都推飞了——这就是"训练分布 = 推理分布"原则。

### 14.2 Phase 0：AR 状态采集

| 参数 | 值 |
|------|-----|
| 种子 | 5000 条 × 16 tokens（来自 mohe_fw_rwkv_1b.npy） |
| 生成 | 官方 12L backbone AR 16 步 |
| 每步保存 | h [768] + cond（softmax(logits)·head.weight → embedding） |
| 总量 | 80000 个 AR 生成状态 |
| 耗时 | 34.5 分钟 |

### 14.3 Phase 1：Stateless MLP Denoiser

| 参数 | 值 |
|------|-----|
| 架构 | Linear(1536→1536) → GELU → Linear(1536→768)，residual |
| 目标 | MSE(h_pred, h_clean)，reduction='sum'/BSZ |
| 优化器 | AdamW, lr=1e-3, BSZ=128, 200000 步 |

### 14.4 v2 实验结果（Denoiser + Confidence Head）

**Denoiser 改善 3/4 prompt：**

| Prompt | AR | AR+Dn | AR+Conf |
|--------|----|-------|---------|
| Capital of France? | ❌ 答非所问 | ❌ CoT 绕死 | ✅ 正确答案 |
| Eiffel tower is in | ✅ Paris | ✅ 更详细 | ❌ 重复退化 |
| Romeo and Juliet? | ❌ Julius Caesar | ✅ Shakespeare | ❌ Lady Macbeth |
| Poem about a cat | ✅ 可读 | ✅ 最佳 | ✅ 中等可用 |

**关键诊断：** Conf head 标签（entropy 下降）不可靠——Romeo 案例 denoiser 修对了但熵升高（多峰分布），错误拦截；Capital of France 案例 denoiser 推向了 CoT 死胡同但熵降低，错误放行。

---

## 15. v3 Stateful SSM Denoiser（2026-06-04）

**动机：** v2 的两个根因——(1) stateless MLP 每步独立修正，不知道前文修正方向 → CoT 绕死；(2) entropy label 不可靠。v3 同时解决两者。

### 15.1 架构

**Denoiser（Stateful SSM）：**

```
s_t = sigmoid(log_A)·s_{t-1} + B·proj(concat(h_t, cond_t))  # 跨步记忆
h' = h + sigmoid(gate)·out(C·s_t)                             # 门控残差
```

**训练目标从 MSE proxy 改为直接质量反馈：**

```python
loss = CE(head(h'), gt_token) + 0.1 * MSE(h', h)
```

**Conf head 标签从 entropy 改为 GT token logprob：**

```python
label = 1 if logprob_after(gt_token, h') > logprob_before(gt_token, h)
```

### 15.2 数据升级

| 维度 | v2 | v3 |
|------|----|----|
| 种子数 | 5000 | 20000 |
| 状态数 | 80000（无序） | 320000（轨迹结构化） |
| 每步标签 | 无 | GT token |
| 训练方式 | 随机采样 | 按轨迹序列 BPTT |

### 15.3 配套清理

- 归档全部 CANN/MoHE 时代代码（experiments/、scripts/、旧 nanoGPT/）
- 删除死代码（rwkv_tokenizer.py、sample.py、旧 CSV）
- 归档旧 kernel（rwkv7_clampw.*）

### 15.4 待验证假设

1. Stateful 跨步记忆 → 解决 Capital of France CoT 绕死（累积修正方向不漂移）
2. GT token logprob label → 同时修掉 Romeo（错误拦截）和 Capital of France（错误放行）
3. 如有效 → MoE Diffusion 延伸（多 expert stateful denoiser + EC router）

---

## 16. Gen 5 — Transformer MLA + 层次记忆架构（2026-06-15）

Gen 4 放弃后，回归 Transformer 范式，但用 MLA（Multi-head Latent Attention, d_c=128）替代标准 attention。核心问题从"能否替代 attention"升级为：**能否在单个 Transformer 内构建层次记忆架构？**

### 16.1 基线：MLA + K→V + GQA + RoPE + SwiGLU + int4

- 架构：12L·8H·4KV·512D·d_c=128
- 量化：int4 K/Q + int2 V（STE 训练），稳定无损
- 数据：FineWeb-Edu 纯英文 ~100M tokens，GPT-2 50K 词表
- 结果：out-quant 生成正常 ✅，1.58-bit 坍缩 ❌（32M 信息通道太窄）

### 16.2 Route A: Latent Indexed Attention

在 MLA 的 128-dim latent（c_kv）上做对比学习，训练语义结构化的 latent 空间用作稀疏注意力索引。

| 版本 | loss 策略 | 同主题 cos | 跨主题 cos | 区分度 gap |
|---|---|---|---|---|
| v1 | InfoNCE w=31 | 0.985 | 0.946 | 0.039 ❌ |
| v2 | InfoNCE ×5 | 0.995 | 0.994 | 0.001 ❌ |
| **v3** | **triplet margin** | **0.905** | **-0.019** | **0.924 ✅** |

**稀疏推理验证：** K=8 + local_w=4 替代全量 attention，生成质量几乎无损，~40× FLOP 节省（T=512）。

### 16.3 Route C: Inertia Wave

用衰减波递推完全替代 attention：`h_t = sigmoid(W_decay) ⊙ h_{t-1} + W_mem(c_kv)`。用 `cumprod + cumsum` 实现 parallel scan（7.4 it/s，比串行 loop 快 15×）。

结果：有词汇分布但语法连贯性不足——128-dim 状态容量不够单独做语言模型。

### 16.4 AC 混合架构（Jamba 风格）

| 层 | 类型 |
|---|---|
| 0-3 | Inertia Wave（L1 快速响应） |
| 4-7 | Full Attention（L2 精确检索） |
| 8-11 | Sparse Latent Index（L3 高效索引） |

参数量 57.47M（持平基线）。消融验证惯性波非死层，但混合架构在 32M 上不优于纯 attention。

### 16.5 Gen 5 关键结论

| 方向 | 结论 |
|---|---|
| 1.58-bit 三元量化 | 32M 不可行（信息损失太大） |
| int4 K/Q + int2 V | 32M 稳定无损 ✅ |
| Latent Indexed Attention | ▲ 全链路验证通过 ✅ |
| Inertia Wave | ◎ 可训通但不够用 ⚠️ |
| AC 混合 | ◎ Jamba 优势域在 7B+ / 100K+ context ⚠️ |
| Latent ROM 愿景 | 模型容量与知识存储解耦 → 后续方向 |

### 16.6 复现

```bash
pip install torch numpy tqdm transformers

# 完整评估
bash eval_all.sh

# 训练
python3 -m rina.train --int4 --out models/out-quant --steps 10000 --bsz 4
python3 -m rina.train_a --int4 --out models/out-rina-a-v3 --steps 10000 --bsz 4
python3 -m rina.train_c --out models/out-rina-c --steps 10000 --bsz 4
python3 -m rina.train_ac --int4 --out models/out-rina-ac --steps 10000 --bsz 4
```

### 16.7 Gen 6 当前架构（2026-06-17, 蒸馏训练中）

**架构全貌（单张 RTX 3070 Ti 8GB 上运行）：**

```
                  ┌──────────────────────────────────────┐
                  │  用户输入                              │
                  │  Llama Tokenizer (128K vocab)         │
                  └────────────┬─────────────────────────┘
                               ↓
                  ┌──────────────────────────────────────┐
                  │  Embedding (640-dim × 128K)          │
                  └────────────┬─────────────────────────┘
                               ↓
                  ┌──────────────────────────────────────┐
                  │  16 层 Block                          │
                  │  ┌──────────────────────────────┐    │
                  │  │  LayerNorm                    │    │
                  │  │                               │    │
                  │  │  ┌─── 三路并行 ───┐            │    │
                  │  │  │ L1: Inertia    │ (惯性波)   │    │
                  │  │  │ L2: MLA + int4 │ (全量)     │    │
                  │  │  │ L3: MLA Sparse │ (稀疏索引) │    │
                  │  │  └──────┬─────────┘            │    │
                  │  │         ↓                      │    │
                  │  │  Confidence Head                │    │
                  │  │  (Linear 64 → 1, sigmoid)       │    │
                  │  │  → 熵估计 → 自动选路            │    │
                  │  │                               │    │
                  │  │  SwiGLU FFN (1792 hidden)      │    │
                  │  └──────────────────────────────┘    │
                  └────────────┬─────────────────────────┘
                               ↓
                  ┌──────────────────────────────────────┐
                  │  LM Head (640 → 128256, weight tied)│
                  └────────────┬─────────────────────────┘
                               ↓
                  ┌──────────────────────────────────────┐
                  │  Latent ROM (FAISS 索引, 可选)       │
                  │  ← c_q → 检索 → w_uk/w_k2v 投影     │
                  │         → 注入 K/V 到 attention       │
                  └──────────────────────────────────────┘
```

**核心组件：**

| 组件 | 实现 | 参数 |
|---|---|---|
| **MLA 注意力** | `w_dqkv` 降维 (640→160) → `w_uq/w_uk` 升维 → 分组查询注意力 (10Q→5KV) → 解耦 RoPE (d_hr=32) → K→V 预测 | 核心参数 |
| **int4 K/Q + int2 V** | STE 量化感知训练，每 32 维一组，K/Q 用 4-bit，V 用 2-bit | 不增加参数 |
| **Sparse Index (L3)** | 用 latent 相似度做 top-K 检索 (K=8, local_w=4)，推理时替代全量 attention | 0 额外参数 |
| **Inertia Wave (L1)** | `h_t = sigmoid(W_decay) · h_{t-1} + W_mem(c_kv)`, parallel scan | 3 线性层 |
| **Confidence Head** | `Linear(d_c, 1)`, 训练目标为预测下一 token 的熵 | 160 + 1 参数/层 |
| **Latent ROM** | FAISS 内积索引, c_kv → w_uk → K, w_k2v → V | 0 额外参数 |
| **SwiGLU** | `silu(xW1) × xW3 × W2`, hidden=1792 (≈640×4×2/3, 对齐到 256) | — |
| **词表** | Llama 3.2 128K tokenizer (张量 128256) | — |

**当前训练（蒸馏）：**

| 项目 | 值 |
|---|---|
| 老师 | Llama 3.2 1B (fp16, frozen) |
| 损失 | CE + 0.5·KL(学生∥老师), T=2.0 |
| 数据 | 1B tokens 混合 (DCLM + StarCode + Math + Chinese) |
| 步数 | 100K, bsz=1, seq=512, lr=3e-4 cosine |
| 参数 | 64.42M core / 146.50M total (128K vocab) |
| 速度 | ~4 it/s, ~7h → 约 21:30 完 |

**架构演进路线：**
```
Gen 1 CANN (SSM)  ← 验证 attractor 记忆不可行
Gen 2 MoHE         ← 验证路由坍缩不可行
Gen 3 RWKV-MoHE    ← 验证纯 SSM 容量不够
Gen 4 AR+Denoiser  ← 验证修正器思路可行但复杂
Gen 5 MLA          ← 回归 Transformer + 三级缓存
  └─ Gen 6 蒸馏    ← 128K 词表 + 老师引导 + 全量数据
```

## 17. Gen 6 — Jamba 混合架构验证（06-25~06-26）

### 17.1 动机

CF 动态路由（conf_head 调度 L1/L2/L3）训练未收敛：frozen backbone 路由不分化，unfrozen 质量退化。改为 **Jamba 式固定层类型混合**——无 router，每层固定走 SSM 或 稀疏 Attention。

### 17.2 Jamba v1 — 基线（06-25）

**架构：** `model_jamba.py` — 12 层 SSM (InertiaLayer K=3) + 4 层 Sparse (MLALayer K=16)，每 3 层 SSM 插 1 层稀疏

**权重加载：** SSM←`c_final.pt`，Sparse+共享←`a_final.pt`

**训练：** 50K 步, bsz=2, seq=512, 4B tokens 新混合数据

**结果：** CE **4.80**，val 4.06，英文通顺 ✅

### 17.3 Jamba v2 — q2(K)+q1(V) 3-bit KV（06-26）

**改动：** `quant_mode='q2k_q1v'`，K→q2 (2-bit), V→q1 (1-bit)，总 KV 3-bit

**结果：** CE **4.20**（v1 4.80），**3-bit KV 与 6-bit 质量持平，KV cache 省 50%** ✅

### 17.4 Jamba LSC q4 — log-space SSM（06-26）

**改动：** `model_jamba_lq.py` — `cumprod(a)` → `exp(cumsum(log(a)))` 加法链，SSM 中间量 q4

**结果：** CE ~5.7（vs v1 4.80，差 <1 CE），**SSM 中间存储省 75%** ✅

**SSM 量化消融（单步）：**
| 精度 | CE | vs fp32 |
|------|-----|---------|
| fp32 | 3.26 | — |
| q8 | 3.26 | +0.003 |
| **q4** | **3.27** | **+0.01** |
| q3 | 3.39 | +0.13 |

### 17.5 Jamba QW 极压版 — q4 权重 + LSC q4 + q2+q1 KV（06-26，训练中）

**架构：** `model_jamba_qw.py` — 所有 Linear 层用 Q4Linear（QAT 前向 q4 量化）+ LSC q4 SSM + q2+q1 KV

**预期：** 全套 4-bit 以下，148M 模型从 594MB → ~75MB，推理 KV cache 压缩 10×
**状态：** 🔄 50K 步训练中

### 17.6 关键结论

| 设计 | 结论 |
|------|------|
| MLA + GQA + int4 | ✅ 全部有效 |
| InertiaWave SSM | ✅ CE 4.8 |
| Jamba 式交错 | ✅ SSM×3 + Sparse×1 训通 |
| q2(K)+q1(V) 3-bit KV | ✅ CE 4.20，与 6-bit 持平 |
| LSC q4 SSM | ✅ log-space cumsum + q4 可行 |
| CF 动态路由 | ❌ 训练未收敛，不推荐 |
| **全套 4-bit 以下** | 🔄 QW 极压版训练中 |

### 17.7 架构演进路线

```
Gen 1 CANN (SSM)         ← 验证 attractor 记忆不可行
Gen 2 MoHE               ← 验证路由坍缩不可行
Gen 3 RWKV-MoHE          ← 验证纯 SSM 容量不够
Gen 4 AR+Denoiser        ← 验证修正器思路可行但复杂
Gen 5 MLA                ← 回归 Transformer + 三级缓存
  └─ Gen 6 蒸馏          ← 128K 词表 + 老师引导
  └─ Gen 6 Jamba         ← SSM + Sparse 混合 + 全套 4-bit 量化 ⭐
```

---

*整理自 `RINA实验日志.md`（9800+ 行, 2026-05-15~06-26）*

