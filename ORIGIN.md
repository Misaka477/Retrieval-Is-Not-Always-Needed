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

RINA is the third architecture. It inherits Natalia's attractor principle, KVR's lesson (don't store, flow), and evolved through three generations in two weeks:

**Gen 1 — CANN-SSM (15M-25M).** Hybrid SSM + attractor with Hebbian plasticity and spike-based gating. It worked — ppl=34.7 on WikiText-103, competitive with GPT-2 at the same size — but scaled poorly. The unified equation `ż = −ε · A(z)` was born here: prediction error as the single driving signal, state and memory coupled through attractor dynamics.

**Gen 2 — MoHE (28M-91M).** First MoE attempt: 4 experts, depth-of-thought iteration. It plateaued at ppl≈3665. The attractor precompression (patterns → field → proj) was sound, but the router never learned to specialize — the residuals were too small, masked by noise and load-balancing.

**Gen 3 — MoHE-RWKV 109M (current).** Switched the SSM backbone to RWKV-v7's WKV linear recurrence (CUDA kernel, head 64, H=12). Replaced the FFN with 12 attractor-based experts, per-token topk=2 routing. A single `torch.stack` replaced a 4608-call Python loop — 54× throughput improvement. Depth-of-thought was fixed from a three-pass no-op to a true iteration chain (h = h_new). ppl=4.9 on a 500M token mix of English, code, math, and Chinese.


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

RINA Gen 3 — MoHE-RWKV 109M (2026-05-29, current)
  ├─ RWKV-v7 WKV kernel (linear recurrence, CUDA, float32)
  ├─ 12 attractor experts, per-token topk=2 routing
  ├─ Batched [12,B,T,D] computation (54× throughput)
  ├─ Depth-3 chain: h = h_new (true iterative refinement)
  ├─ 500M tokens: FW + StarCoder + Math + Chinese
  ├─ ppl=4.9, train ppl=4.9, val ppl=4.3, route entropy=0.5

```


## On Hardware

RINA is developed on a single ROG Zephyrus M16 2022 (RTX 3070 Ti Laptop GPU, 8 GB VRAM). The 109M MoHE-RWKV model — 12-expert MoE with CUDA WKV kernel, per-token routing, depth-3 chain — runs at 3 it/s @ SEQ=512 with batch 4. The entire 90000-step training takes ~12.5 hours.

The physical constraints — 8 GB VRAM ceiling, single consumer GPU — forced every architectural decision. The batched attractor (one torch.stack instead of 4608 Python calls) was not an optimization; it was the only way to reach production speed. The per-token routing was not a design preference; it was the only way to make MoE work without collapse. Constraints are not obstacles to work around. They are the shape of the solution.


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

*Written 2026-05-19, 04:20. Updated 2026-05-30, after 90000 steps of MoHE-RWKV 109M: ppl=4.9, route differentiated, ~2 hours to run 30000 steps.*

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

RINA 是第三代架构。它继承了 Natalia 的吸引子原理和 KVR 的教训（别存，流），在三周内迭代了三代：

**Gen 1 — CANN-SSM (15M-25M)。** SSM + 吸引子 + Hebbian。能在 WikiText-103 上跑出 ppl=34.7，同一参数量下匹敌 GPT-2，但规模上不去。统一公式 `ż = −ε · A(z)` 在这一代成形。

**Gen 2 — MoHE (28M-91M)。** 首个 MoE 尝试：4 专家 + 深度迭代。卡在 ppl≈3665。吸引子的思路是对的，但路由因残差太小、噪声和负载均衡压制而无法分化。

**Gen 3 — MoHE-RWKV 109M（当前）。** 将 SSM 骨干替换为 RWKV-v7 的 WKV 线性递回（CUDA kernel, head 64, H=12）。FFN 替换为 12 专家吸引子 + per-token topk=2 路由。一次 `torch.stack` 替代了 4608 次 Python 调用 —— 54× 吞吐。depth-of-thought 从空转修正为真迭代链（h = h_new）。ppl=4.9，500M token 中英代码混合数据，一台 ROG Zephyrus M16 2022。


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

RINA Gen 3 — MoHE-RWKV 109M (2026-05-29, 当前)
  ├─ RWKV-v7 WKV 内核 (线性递回, CUDA, float32)
  ├─ 12 专家吸引子 + per-token topk=2 路由
  ├─ 批量化 [12,B,T,D] 计算 (54× 吞吐)
  ├─ Depth-3 真链: h = h_new
  ├─ 500M tokens: FW + StarCoder + Math + 中文
  ├─ ppl=4.9, 路由熵=0.5

```


## 硬件

RINA 在一台 ROG Zephyrus M16 2022（RTX 3070 Ti Laptop GPU，8 GB VRAM）上开发。109M 的 MoHE-RWKV 模型 —— 12 专家 MoE + CUDA WKV kernel + per-token 路由 + depth-3 链 —— 以 3 it/s @ SEQ=512、batch=4 运行。90000 步训练约需 12.5 小时。

8 GB 显存天花板和单张消费级显卡驱动了每一个架构决策。批量化 attractor（一次 torch.stack 替代 4608 次 Python 调用）不是优化，是在 8G 卡上跑出生产速度的唯一路径。per-token 路由不是设计偏好，是让 MoE 不坍缩的唯一方式。约束不是绕过去的障碍。约束是解的模具。


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

*2026年5月20日 04:56。更新于 2026年5月30日。MoHE-RWKV 109M 已在 90000 步后达到 ppl=4.9，路由已分化，约 2 小时跑完 30000 步。*
