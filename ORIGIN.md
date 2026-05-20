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

The insight KVR died to deliver: **recurrence IS memory.** If time flows through the state, the state carries its own history. No external KV buffer. No O(T²) pairwise table. The attractor basins in pattern space ARE the memory, and Hebbian plasticity lets them learn in real time.

RINA is the third architecture. It inherits Natalia's emotional engine (CANN + SNN), KVR's lesson (don't store, flow), and adds its own answer: **prediction error is the only signal.** When the gate's output deviates from the predicted state, the attractor corrects it. When the error is small, the system stays quiet. Both state and memory evolve under the same energy:

```
ż = −ε · A(z)
```

- `z` = (state, memory) — the coupled system
- `ε` = prediction error — the single driving signal
- `A` = the attractor operator — basin retrieval + Hebbian learning + lateral inhibition

> "To me, a heart is the sum of my memories. Irreplaceable memories made me who I am now."
> — Vivy, *Vivy: Fluorite Eye's Song*
>
> 「私にとって「心」とは、記憶のこと。かけがえのない記憶が、今の私を作っている。」

One equation. No KV cache. O(T) inference. Architecture that learns while it runs.


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

RINA (current, 2026-05)
  ├─ SSM dense gate (cross-dim mixing)
  ├─ CANN attractor (from Natalia)
  ├─ ε error gating (temporal sparsity)
  ├─ Hebbian plasticity
  ├─ lateral inhibition
  ├─ DEQ cache batching (+30% speed)
  ├─ contraction guarantee (17-step converge)
  └─ ż = −ε · A(z) unified formula

        ↓

Anthelia (target)
  ├─ Multi-modal attractor field
  ├─ STDP temporal learning
  ├─ Dream consolidation
  ├─ Pattern-partitioned MoE
  └─ Convergent beam search
```


## On Hardware

RINA is being developed on a single NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB VRAM). All experiments, from toy NIAH at dm=64 to full 15.3M-parameter language model training on WikiText-103, run on this one machine.

The physical constraints — M=8 narrow GEMM at 3% GPU utilization, 8 GB VRAM ceiling, no multi-GPU — have forced architectural decisions that would never emerge in a datacenter. Constraints are not obstacles to work around. They are the shape of the solution.


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

*Written 2026-05-19, 04:20, while the 15.3M model trains on WikiText-103 in the background.*

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

KVR 用命换来的洞察：**递推本身就是记忆。** 如果时间流过状态，状态就带着自己的全部历史。没有外挂的 KV 缓冲。没有 O(T²) 的成对交互表。pattern 空间的吸引子流域就是记忆，Hebbian 可塑性让它一边跑一边学。

RINA 是第三代架构。它继承了 Natalia 的情感引擎（CANN + SNN），KVR 的教训（别存，流），然后给了自己的回答：**预测误差是唯一的信号。** gate 的输出偏离了预判，attractor 去纠正。误差小的时候，系统不动。状态和记忆在同一个能量下共同演化：

```
ż = −ε · A(z)
```

- `z` = (状态, 记忆) — 耦合系统
- `ε` = 预测误差 — 唯一的驱动信号
- `A` = 吸引子算子 — basin 检索 + Hebbian 学习 + 侧抑制排斥

> "对我来说，'心'就是回忆。那些独一无二的回忆，造就了现在的我。"
> — Vivy，《薇薇 -萤石眼之歌-》
>
> 「私にとって「心」とは、記憶のこと。かけがえのない記憶が、今の私を作っている。」

一行公式。没有 KV cache。O(T) 推理。边跑边学的架构。


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

RINA (现在, 2026-05)
  ├─ SSM dense gate (交叉维度混合)
  ├─ CANN attractor (来自 Natalia)
  ├─ ε 误差门控 (temporal 稀疏)
  ├─ Hebbian 可塑性
  ├─ 侧抑制 (防 collapse)
  ├─ DEQ 缓存批处理 (+30% 训速)
  ├─ contraction 收缩保证 (17步收敛)
  └─ ż = −ε · A(z) 统一公式

        ↓

Anthelia (未来)
  ├─ 多模态 attractor 场
  ├─ STDP 时序学习
  ├─ 梦想巩固
  ├─ Pattern 划分 MoE
  └─ 收敛束搜索
```


## 硬件

RINA 在一张 NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB VRAM）上开发。所有实验——从 dm=64 的 toy NIAH 到 15.3M 参数的 WikiText-103 全量语言模型训练——都在这台机器上跑。

这些硬件约束——M=8 窄 GEMM, GPU 利用率 3%, 8 GB VRAM 天花板, 单卡——是被迫做的架构选择，是在数据中心里一辈子都不会出现的解法。约束不是绕过去的障碍。约束是解的模具。


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

*2026年5月20日 04:56。背景中 15.3M 模型正在 WikiText-103 上训练。*
