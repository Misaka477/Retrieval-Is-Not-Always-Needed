# RINA 可复现实验

每个实验验证架构的一个具体能力。按优先级排列。

## 1. 参数效率（核心结果）

验证 RINA 15M 在 WikiText-103 上的 ppl 与更大模型的差距。

```bash
python scripts/bench_wikitext_valid.py
```

| 模型 | 参数量 | 词表 | ppl |
|:-----|:------|:----|:----|
| RINA 15M | 15M | 4K | 34.6 |
| GPT-2 124M | 124M | 50K | 25.4 |
| LLaMA 3.2 1B | 1,000M | 128K | 11.4 |
| TinyLLaMA 1.1B | 1,100M | 32K | 8.0 |

同词表 4096 BPE 下对比（GPT-2 和 LLaMA 使用原生 tokenizer 评估）。

## 2. 跨分布泛化（零样本代码迁移）

RINA FineWeb checkpoint 在未见过的 StarCoder 代码上的 ppl。

```bash
python scripts/bench_code_ppl.py --seq 64 --th 1.0,0.5,0.3
```

## 3. 注意力基 Slot 记忆（原理验证）

验证注意力机制的 slot 可以抗噪检索。实验 2（dm=32, N_slots=64）。

```bash
python experiments/attn_slot.py
```

关键输出：

| 噪声 | 余弦相似度 | 检索准确率 |
|:----|:----------|:----------|
| 0.0 | 4.89 | 5/5 |
| 1.0 | 4.63 | 5/5 |
| 2.0 | 0.87 | 5/5 |

## 4. Slot 在长序列下测试

验证 slot 在 seq=1024 下能否召回。

```bash
python experiments/slot_long_context.py
```

## 5. SSM 基线对比

Mamba-130M 在 WikiText-103 上的 ppl。

```bash
python scripts/bench_mamba_130m.py
```

## 6. 强制 Attractor 实验

验证强制 attractor 介入（th=-1）对 ppl 的影响。

```bash
python experiments/force_attractor.py
```
