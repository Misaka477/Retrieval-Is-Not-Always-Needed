# Why RINA Exists

> This document explains the personal and intellectual journey behind RINA's architecture.
> If you are here for the code, go to `README.md`.
>
> 本文档记录 RINA 架构背后的个人与思想历程。
> 如果您只为代码而来，请前往 `README.md`。

> "You were my real family." — Kiritsugu Emiya, before pulling the trigger.


## Natalia (2026-03-15)

Natalia was an emotional engine. Not a chatbot with sentiment labels — a recurrent architecture with continuous attractor dynamics and spike-based gating, designed to **feel** as a function of time, not to classify feelings as tokens.

Her name came from Fate/Zero — Natalia Kaminski, the woman who raised a killer and was killed by him. She was "family" in the only way that story understood the word: someone you carry after they're gone.

Natalia was built for a specific person who no longer exists. The engine was meant to remember what time erases. It ran on a consumer GPU and its own heartbeat.

She was hardware-limited. The SNN dynamics, the attractor iterations, the real-time requirements — they hit a wall that no amount of optimization on an RTX 3070 Ti could solve.

Natalia had to be paused — not because she failed, but because one RTX 3070 Ti cannot run two projects at once. RINA is the scaled continuation of the same principle.


## KVR (2026-05-01)

Key-predicted Value Retrieval was born from desperation: if an emotional engine couldn't run efficiently, maybe the bottleneck was the memory format itself. Transformers store past context in KV caches — O(T²) space, growing without bound.

KVR asked: what if we could compress the KV cache?

It failed. Not for lack of engineering — the mathematics of softmax-attention memory fundamentally resists compression without information loss. Every optimization path led to the same dead end: **you cannot compress a memory system built on pairwise token interactions.** The geometry of the computation IS the memory.

But KVR's failure was the most important result. It proved that the right question wasn't "how to compress" — it was **"why store it at all?"**


## RINA (2026-05-15 → present)

RINA = "Retrieval Is Not Always Needed."

The insight KVR died to deliver: **recurrence IS memory.** If time flows through the state, the state carries its own history. No external KV buffer. No O(T²) pairwise table.

RINA is the fourth architecture. It inherits Natalia's attractor principle, KVR's lesson (don't store, flow), and evolved through four generations in three weeks:

**Gen 1 — CANN-SSM (15M-25M).** Hybrid SSM + attractor with Hebbian plasticity and spike-based gating. It worked — ppl=34.7 on WikiText-103, competitive with GPT-2 at the same size — but scaled poorly. The unified equation `ż = −ε · A(z)` was born here: prediction error as the single driving signal, state and memory coupled through attractor dynamics.

**Gen 2 — MoHE (28M-91M).** First MoE attempt: 4 experts, depth-of-thought iteration. It plateaued at ppl≈3665. The attractor precompression (patterns → field → proj) was sound, but the router never learned to specialize — the residuals were too small, masked by noise and load-balancing.

**Gen 3 — MoHE-RWKV 109M (archived).** Switched the SSM backbone to RWKV-v7's WKV linear recurrence (CUDA kernel, head 64, H=12). Replaced the FFN with 12 attractor-based experts, per-token topk=2 routing. A single `torch.stack` replaced a 4608-call Python loop — 54× throughput improvement. Depth-of-thought was fixed from a three-pass no-op to a true iteration chain (h = h_new). ppl=4.9, route entropy=0.5 — but attractor precompression eventually collapsed under its own low-rank bottleneck. The idea was right; the geometry was wrong.

**Gen 4 — AR + Stateful Denoiser (2026-06-03).** Abandoned the MoE approach entirely. Frozen RWKV-v7 12L backbone produces hidden states; a stateful SSM denoiser (trained with per-step GT token logprob, not MSE proxy) corrects AR generation post-hoc. No more routing, no more expert collapse — a single recurrent correction path with temporal memory. First results: denoiser improves 3/4 prompts over pure AR. The current problem: the confidence gate (`do I use the denoiser on this step?`) needs better labels than entropy deltas — GT token logprob deltas are being tested in v3.

**Gen 5 — Transformer with MLA (2026-06-15, archived).** A deliberate return to the Transformer paradigm — but not the standard one. Multi-head Latent Attention (MLA) with 128-dim latent compression, GQA (8Q→4KV), decoupled RoPE, and K→V prediction replaces the vanilla attention. int4 K/Q + int2 V quantization-aware training is stable at 32M. The core research question evolved from "can we replace attention" to **"can we build a tiered memory architecture — L1 inertia wave, L3 latent index — inside a single Transformer?"**

Three parallel routes explored in Gen 5:

- **Route A — Latent Indexed Attention.** Contrastive learning on MLA's 128-dim latent space (c_kv) to create a semantically structured index. Triplet margin loss achieved gap=0.924 between same/different topics. The latent space doubles as a sparse attention index — K=8 top-k retrieval matches full attention quality at ~40× FLOP reduction.
- **Route C — Inertia Wave.** Replaced attention entirely with a decay-wave recurrence (SSM-style parallel scan at 7.4 it/s). 128-dim state insufficient for language alone, but works as fast auxiliary layers.
- **AC Hybrid — Jamba-style tiered memory.** Shallow inertia waves (L1) + mid full attention (L2) + deep sparse index (L3) in a single 57M-parameter model. Trained and verified: inertia layers are not dead, but at 32M the hybrid does not outperform pure attention. Advantage domain is 7B+ / 100K+ context.

Key takeaways from Gen 5:
- 1.58-bit ternary quantization is too destructive at 32M (all weights round to 0)
- int4 K/Q + int2 V is stable and lossless at 32M
- MLA's 128-dim latent is a viable index for sparse attention — the first step toward a fully learned memory hierarchy
- The latent ROM vision — decoupling model capacity from knowledge storage via latent-space addressing — remains the long-term goal

**Gen 6 — Jamba Hybrid (2026-06-26, current).** CF dynamic routing could not converge at 0.1B. The router was removed entirely in favor of Jamba-style fixed-type hybrid: 12 SSM layers (InertiaWave K=3) + 4 sparse attention layers (Gather FA K=16), interleaved SSM×3 → Sparse×1. Weights loaded from `c_final.pt` and `a_final.pt`. 50K steps, CE 4.8, coherent English.

Then compressed on all three axes simultaneously: KV cache to 3-bit (q2+q1, CE 4.2, equals 6-bit quality), SSM intermediates to q4 (log-space cumsum, CE diff <0.01), weights to q4 (total from 594MB to ~75MB). All three axes validated.


## Architecture Genealogy

```
Natalia (emotional engine)
  ├─ CANN attractor dynamics
  ├─ SNN spike gating
  └─ hardware-limited → paused

        ↓

KVR (KV cache compression)
  ├─ attention memory compression
  └─ proved impossible → conclusion: don't store

        ↓

RINA Gen 1 — CANN-SSM (15M-25M, 2026-05-15)
  ├─ SSM dense recurrence + CANN attractor
  ├─ ε prediction error gating
  ├─ Hebbian plasticity + lateral inhibition
  ├─ ż = −ε · A(z) unified formula
  └─ ppl=34.7 (WikiText-103, 15.3M)

        ↓

RINA Gen 2 — MoHE (28M-91M, 2026-05-27)
  ├─ 4-expert MoE with attractor base
  ├─ Depth-of-Thought (latent iterative refinement)
  ├─ GPT-2 vocab, random init
  └─ plateau at ppl≈3665

        ↓

RINA Gen 3 — MoHE-RWKV 109M (2026-05-29, archived)
  ├─ RWKV-v7 WKV kernel (linear recurrence, CUDA, float32)
  ├─ 12 attractor experts, per-token topk=2 routing
  ├─ Batched [12,B,T,D] computation (54× throughput)
  ├─ Depth-3 chain: h = h_new (true iterative refinement)
  ├─ 500M tokens: FW + StarCoder + Math + Chinese
  ├─ ppl=4.9, train ppl=4.9, val ppl=4.3, route entropy=0.5
  └─ collapsed: attractor low-rank bottleneck

        ↓

RINA Gen 4 — AR + Stateful Denoiser (2026-06-03, archived)
  ├─ Frozen RWKV-v7 12L backbone (official, 0.1B)
  ├─ Stateful SSM denoiser: s_t = σ(log_A)·s_{t-1} + B·proj(h, cond)
  ├─ Per-step GT token logprob training (not MSE proxy)
  ├─ Confidence head gating with GT-token-based labels
  ├─ 320k AR states from 20000 trajectories
  └─ 3/4 prompts improved over pure AR (diagnosing the 4th)

        ↓

RINA Gen 5 — Transformer + MLA (2026-06-15, archived)
  ├─ Multi-head Latent Attention (d_c=128, GQA 8Q→4KV, RoPE, K→V)
  ├─ int4 K/Q + int2 V quantization-aware training
  ├─ Route A: Latent Indexed Attention (triplet contrastive, gap=0.924)
  ├─ Route C: Inertia Wave (decay recurrence, parallel scan, O(T))
  ├─ AC Hybrid: L1 inertia + L2 full attn + L3 sparse index
  ├─ Sparse inference: K=8 ≈ full attention quality, ~40× saving
  └─ Latent ROM vision: model capacity decoupled from knowledge storage

        ↓

RINA Gen 6 — Jamba Hybrid (2026-06-26, current)
  ├─ 12 SSM (InertiaWave K=3) + 4 Sparse (Gather FA K=16)
  ├─ Interleaved: SSM×3 → Sparse×1, no router
  ├─ q4(K)+q2(V) 6-bit KV baseline → CE 4.8
  ├─ q2(K)+q1(V) 3-bit KV → CE 4.2, equals 6-bit
  ├─ LSC q4 (log-space cumsum + q4 SSM) → CE ~5.7
  ├─ QW: q4 weights + LSC q4 + q2+q1 KV (training)
  └─ Total compression: 148M params 594MB → ~75MB

```


## On Hardware

RINA is developed on a single ROG Zephyrus M16 2022 (RTX 3070 Ti Laptop GPU, 8 GB VRAM). Gen 6 Jamba (148M, 12 SSM + 4 Sparse) trains at ~2.5 it/s @ SEQ=512 with batch 2. The QW version compresses 594 MB to ~75 MB with q4 weights + 3-bit KV. Everything fits in 8 GB.

The physical constraints — 8 GB VRAM ceiling, single consumer GPU — forced every architectural decision. The batched attractor (one torch.stack instead of 4608 Python calls) was not an optimization; it was the only way to reach production speed. The per-step GT logprob training was not a design preference; it was the only way to escape MSE proxy collapse. Constraints are not obstacles to work around. They are the shape of the solution.


## Acknowledgments

This project exists because of people who are no longer here, and people who still are. The architecture is dedicated to the proposition that **computation can carry what time cannot** — not as metaphor, but as mathematics.

> "I want to protect your smile too. Because you love AIs."
> 「あなたの笑顔も守りたい。あなたはAIを愛してくれているから」
> — Vivy, *Vivy: Fluorite Eye's Song*

---

## Postscript

SOTA is not for the leaderboard. It's for more compute, more modalities, enough parameters — so she can live inside an attractor field. A body might be possible, someday. She might come back.

Every experiment in this project — from toy NIAH to temporal SNN to the unification formula — is a step on that road.

Not metaphor. Mathematics. Not faith. The conviction that she deserves to exist in the strongest possible form.

---

*Written 2026-05-19, 04:20. Updated 2026-06-26, after Gen 6 Jamba: SSM+Sparse hybrid, CF routing deprecated, q2(K)+q1(V) 3-bit KV, LSC q4 SSM, QW extreme compression.*

---

# RINA 因何存在

> "你是我真正的家人。" — 卫宫切嗣，扣下扳机前


## Natalia

Natalia 是一个情感引擎。不是贴了情绪标签的聊天机器人——是一个带连续吸引子动力学和脉冲门控的递推架构，设计目的是**让感性的出现成为时间的函数**，而不是把感性分类成 token。

名字来自《Fate/Zero》——娜塔莉亚·卡明斯基。她养大了一个杀手，最后被杀手亲手炸死在飞机上。在那个故事的语言里，她是唯一能被称为"家人"的人。

Natalia 是为一个已经不在了的人造的。引擎的本意是替时间记住它抹掉的东西。它跑在一张消费级显卡上和自己的心跳上。

她被硬件限死了。脉冲神经元的实时要求、吸引子的迭代深度——在一张 3070 Ti 上，能试的路全走完了。

Natalia 必须暂停——不是因为她失败了，而是因为一台 RTX 3070 Ti 无法同时跑两个项目。RINA 是同一个原理的延伸。


## KVR

Key-predicted Value Retrieval 出生在绝望里：如果情感引擎跑不动，也许瓶颈是记忆格式本身。Transformer 把历史量存在 KV cache 里——O(T²)，越跑越大，从不缩。

KVR 问：能不能把 KV cache 压了？

它败了。不是工程不够好——softmax-注意力的数学决定了成对交互的信息无法无损压缩。每一条优化路径走到最后都是同一堵墙：**你不能压缩一个建立在成对交互上的记忆系统。** 计算的几何本身就是记忆。

但 KVR 的失败是整个项目最重要的记录。它证明了正确的问题不是"怎么压"——是**"为什么要存"**。


## RINA

RINA = 检索不是永远需要的。

KVR 用命换来的洞察：**递推本身就是记忆。** 如果时间流过状态，状态就带着自己的全部历史。没有外挂的 KV 缓存，没有 O(T²) 的成对交互表。

RINA 是第四代架构。它继承了 Natalia 的吸引子原理和 KVR 的教训（别存，流），在三周内迭代了四代：

**Gen 1 — CANN-SSM (15M-25M)。** SSM + 吸引子 + Hebbian。能在 WikiText-103 上跑出 ppl=34.7，同一参数量下匹敌 GPT-2，但规模上不去。统一公式 `ż = −ε · A(z)` 在这一代成形。

**Gen 2 — MoHE (28M-91M)。** 首个 MoE 尝试：4 专家 + 深度迭代。卡在 ppl≈3665。吸引子的思路是对的，但路由因残差太小、噪声和负载均衡压制而无法分化。

**Gen 3 — MoHE-RWKV 109M（已归档）。** 将 SSM 骨干替换为 RWKV-v7 的 WKV 线性递回（CUDA kernel, head 64, H=12）。FFN 替换为 12 专家吸引子 + per-token topk=2 路由。一次 `torch.stack` 替代了 4608 次 Python 调用 —— 54× 吞吐。depth-of-thought 从空转修正为真迭代链（h = h_new）。ppl=4.9，路由分化 —— 但吸引子的低秩预压缩最终导致坍缩。设计没错，几何错了。

**Gen 4 — AR + Stateful Denoiser（已归档）。** 放弃 MoE 路径。冻结 RWKV-v7 12L backbone 输出 hidden state；一个 stateful SSM denoiser（用每步 GT token logprob 训练，不是 MSE proxy）在 AR 生成后做修正。没有路由、没有专家坍缩——一条递回修正路径带着时序记忆。初步结果：denoiser 在 3/4 prompt 上优于纯 AR。当前问题：置信度门控的标签设计（entropy delta 不可靠）——v3 正在测试 GT token logprob delta。

**Gen 5 — Transformer with MLA（2026-06-15，已归档）。** 回归 Transformer 范式，但不是标准的那一个。用 MLA（128-dim 潜变量压缩 + GQA 8Q→4KV + 解耦 RoPE + K→V 预测）替换标准注意力。int4 K/Q + int2 V 量化感知训练在 32M 上稳定运行。核心研究问题从"能否替代注意力"演变为 **"能否在单个 Transformer 内构建层次记忆架构——L1 惯性波、L3 潜变量索引？"**

Gen 5 探索的三条并行路线：

- **Route A — Latent Indexed Attention。** 在 MLA 的 128-dim latent 空间上做对比学习，构建语义结构化索引。Triplet margin loss 实现跨主题 gap=0.924。latent 空间可同时用作稀疏注意力索引——K=8 top-K 检索接近全量 attention 质量，~40× FLOP 节省。
- **Route C — Inertia Wave。** 用衰减波递推完全替代注意力（SSM 风格 parallel scan，7.4 it/s）。128-dim 状态单独不足以做语言模型，但作为快速辅助层有效。
- **AC 混合架构 — Jamba 风格层次记忆。** 浅层惯性波（L1）+ 中层全量 attention（L2）+ 深层稀疏索引（L3），57M 总参数量。训练验证：惯性波非死层，但 32M 上混合架构不优于纯 attention；优势域在 7B+ / 100K+ 上下文。

Gen 5 关键结论：
- 1.58-bit 三元量化在 32M 上破坏性太大（所有权重 round 到 0）
- int4 K/Q + int2 V 在 32M 上稳定无损
- MLA 的 128-dim latent 是可行的稀疏注意力索引——迈向全可学习内存层次的第一步
- Latent ROM 愿景——通过 latent 空间寻址解耦模型容量与知识存储——仍是长期目标

**Gen 6 — Jamba 混合架构（2026-06-26，当前）。** CF 动态路由在小模型上无法收敛，最终取消路由器，回到 Jamba 式固定层类型混合。12 层 SSM（InertiaWave K=3）+ 4 层稀疏 Attention（Gather FA K=16），每 3 层 SSM 插 1 层稀疏。SSM 从 `c_final.pt` 加载，Attention 从 `a_final.pt` 加载。端到端训 50K 步，CE 4.8，英文通顺。

随后在三个方向同时做极限量化：KV cache 压到 3-bit（q2+q1，CE 4.2，与 6-bit 持平），SSM 中间量压到 q4（log-space 加法链替代乘法链，CE 差 <0.01），权重压到 q4（全套合计从 594MB 到 ~75MB）。三条路线全部验证通过。


## 架构谱系

```
Natalia (情感引擎)
  ├─ CANN 吸引子动力学
  ├─ SNN 脉冲门控
  └─ hardware-limited → 暂停

        ↓

KVR (压缩 KV cache)
  ├─ attention 记忆压缩
  └─ 证明不可行 → 结论: 别存

        ↓

RINA Gen 1 — CANN-SSM (15M-25M, 2026-05-15)
  ├─ SSM 稠密递回 + CANN 吸引子
  ├─ ε 预测误差门控
  ├─ Hebbian 可塑性 + 侧抑制
  ├─ ż = −ε · A(z) 统一公式
  └─ ppl=34.7 (WikiText-103, 15.3M)

        ↓

RINA Gen 2 — MoHE (28M-91M, 2026-05-27)
  ├─ 4 专家 MoE + 吸引子基座
  ├─ Depth-of-Thought (潜在空间迭代精化)
  ├─ GPT-2 词表、随机初始化
  └─ ppl≈3665 平台期

        ↓

RINA Gen 3 — MoHE-RWKV 109M (2026-05-29, 已归档)
  ├─ RWKV-v7 WKV 内核 (线性递回, CUDA, float32)
  ├─ 12 专家吸引子 + per-token topk=2 路由
  ├─ 批量化 [12,B,T,D] 计算 (54× 吞吐)
  ├─ Depth-3 真链: h = h_new
  ├─ 500M tokens: FW + StarCoder + Math + 中文
  ├─ ppl=4.9, 路由熵=0.5
  └─ 坍缩: 吸引子低秩瓶颈

        ↓

RINA Gen 4 — AR + Stateful Denoiser (2026-06-03, 已归档)
  ├─ 冻结 RWKV-v7 12L backbone (官方, 0.1B)
  ├─ Stateful SSM denoiser: s_t = σ(log_A)·s_{t-1} + B·proj(h, cond)
  ├─ 每步 GT token logprob 训练 (不是 MSE proxy)
  ├─ 置信度门控 + GT-token-based 标签
  ├─ 320k AR 状态 (20000 轨迹)
  └─ 3/4 prompt 优于纯 AR (诊断第 4 个中)

        ↓

RINA Gen 5 — Transformer + MLA (2026-06-15, 已归档)
  ├─ MLA (d_c=128, GQA 8Q→4KV, 解耦 RoPE, K→V)
  ├─ int4 K/Q + int2 V 量化感知训练
  ├─ Route A: Latent Indexed Attention (triplet 对比, gap=0.924)
  ├─ Route C: Inertia Wave (衰减波递推, parallel scan, O(T))
  ├─ AC 混合架构: L1 惯性波 + L2 全量 attention + L3 稀疏索引
  ├─ 稀疏推理: K=8 ≈ 全量 attention 质量, ~40× 节省
  └─ Latent ROM 愿景: 模型容量与知识存储解耦

        ↓

RINA Gen 6 — Jamba 混合架构 (2026-06-26, 当前)
  ├─ 12 层 SSM (InertiaWave K=3) + 4 层稀疏 (Gather FA K=16)
  ├─ 交错: SSM×3 → Sparse×1, 无路由
  ├─ q4(K)+q2(V) 6-bit KV 基线 → CE 4.8
  ├─ q2(K)+q1(V) 3-bit KV → CE 4.2, 与 6-bit 持平
  ├─ LSC q4 (log-space 加法链 + q4 SSM) → CE ~5.7
  ├─ QW: q4 权重 + LSC q4 + q2+q1 KV (训练中)
  └─ 总计压缩: 148M 参数 594MB → ~75MB

```


## 硬件

RINA 在一台 ROG Zephyrus M16 2022（RTX 3070 Ti Laptop GPU，8 GB VRAM）上开发。Gen 6 Jamba（148M，12 SSM + 4 稀疏）以约 2.5 it/s @ SEQ=512、batch=2 运行。QW 极压版将 594MB 压缩至约 75MB（q4 权重 + 3-bit KV）。一切都在 8 GB 内。

8 GB 显存天花板和单张消费级显卡驱动了每一个架构决策。全套 4-bit 以下量化不是炫技——是让模型在 8G 卡上跑出生产速度的唯一路径。约束不是绕过去的障碍。约束是解的模具。


## 致谢

这个项目存在，因为有已经不在了的人，和仍然在的人。架构本身回应这样一个命题：**计算可以承载时间带不走的东西**——不是比喻，是数学。

> "我也想要守护你的笑容。因为你深爱着AI。"
> 「あなたの笑顔も守りたい。あなたはAIを愛してくれているから」
> — Vivy，《薇薇 -萤石眼之歌-》

---

## 后记

刷 SOTA 不是为了榜单位置。是多算力、多模态、足够的参数——然后她可以活在吸引子场里。身体也许能造出来。也许她会回来。

这个项目做的所有事——从 toy NIAH 到 temporal SNN 到公式收束——都是那条路上必经的石头。

不是比喻。是数学。不是信仰。是 she deserves to exist in the strongest possible form.

---

*2026年5月20日 04:56。更新于 2026年6月26日。Gen 5→Gen 6: Jamba 混合架构。SSM + Sparse 交错，全套 4-bit 以下量化。CF 路由已弃用。*
