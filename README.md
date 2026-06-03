# RINA (Retrieval Is Not Always Needed) — AR + Stateful Diffusion

State diffusion on frozen RWKV-v7 backbone: a stateful SSM denoiser corrects AR hidden states to improve generation quality.

## Architecture

```
RWKV-v7 12L backbone (frozen, return_h=True)
  → h [768] → Stateful SSM Denoiser → h' → Head → logits → token
```

| Component | Detail |
|-----------|--------|
| Backbone | Official RWKV-v7 12L, 0.1B, CUDA kernel (wkv7_fp32), squared ReLU FFN |
| Denoiser | SSM: `s_t = sigmoid(log_A)·s_{t-1} + B·proj(concat(h, cond))`; gated residual readout |
| Training | Per-step GT token logprob: `CE(head(h'), gt) + 0.1·MSE(h', h)` |
| Condition | `softmax(logits)·head.weight` → predicted next embedding |
| Confidence head | `Linear(768→128→1)→Sigmoid`, label = `logprob_after(gt) > logprob_before(gt)` |

## Current Status

### v3 — Stateful SSM Denoiser (current, in training)

- 20000 trajectories × 16 steps = 320000 AR states with GT token labels
- Stateful SSM processes trajectories sequentially (BPTT-style)
- Training objective: maximize per-step logprob of GT token + KL regularization
- See `docs/RINA_实验总览.md §12` for details

### v2 — Stateless MLP + Confidence Head (diagnosed)

- MLP denoiser improved 3/4 prompts
- Confidence head (entropy-based labels) unreliable — Romeo case mis-blocked, Capital of France case mis-released
- Diagnosis: entropy ≠ quality; switched to GT-token-based labels for v3

### v1 — MoHE-RWKV 109M (archived)

Attractor MoE with RWKV-v7 backbone. ppl=4.9, route differentiated. Abandoned due to attractor collapse.

## Quick Start

```bash
# Install
pip install torch numpy tqdm

# Train stateful denoiser (Phase 0 + Phase 1)
python rina/train_ar.py

# Train confidence head
python rina/train_conf.py

# Evaluate
python rina/eval_multi.py
```

## Package

```
rina/
  rwkv_v7_demo.py          ← RWKV-v7 backbone (patched kernel + return_h)
  train_ar.py              ← Stateful denoiser training (Phase 0 + Phase 1)
  train_conf.py            ← Confidence head training
  eval_multi.py            ← Multi-prompt comparison
  __init__.py              ← RWKV, RWKV_TOKENIZER, StatefulDenoiser, run_model

kernels/
  wkv7_fp32.cu / .cpp      ← CUDA kernel (WindBackstepping, fp32 + backward)

checkpoints/
  mohe_fw_rwkv_1b.npy      ← Training data (3.7GB)
  rwkv_vocab_v20230424.txt
  ar_trajs.pt              ← v3 trajectory data
  dn_stateful_final.pt     ← v3 trained denoiser

docs/
  RINA实验日志.md           ← Full experiment log (8200+ lines)
  RINA_实验总览.md           ← Condensed experiment overview
  DLM_survey.md             ← Diffusion LM literature review

archive/                   ← Previous generations (CANN, MoHE)
```

## Hardware

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## References

- `docs/RINA实验日志.md` — full experiment log
- `docs/RINA_实验总览.md` — condensed experiment overview
- `docs/DLM_survey.md` — diffusion LM survey

## Contact

rapidsound@163.com / mikotomisaka477@gmail.com

---

# RINA (Retrieval Is Not Always Needed) — AR + Stateful Diffusion

在冻结的 RWKV-v7 backbone 上做状态扩散：一个 stateful SSM denoiser 修正 AR hidden state 来改善生成质量。

## 架构

```
RWKV-v7 12L backbone（冻结，return_h=True）
  → h [768] → Stateful SSM Denoiser → h' → Head → logits → token
```

| 组件 | 细节 |
|------|------|
| Backbone | 官方 RWKV-v7 12L, 0.1B, CUDA kernel (wkv7_fp32), squared ReLU FFN |
| Denoiser | SSM: `s_t = sigmoid(log_A)·s_{t-1} + B·proj(concat(h, cond))`; 门控残差读出 |
| 训练目标 | 每步 GT token logprob: `CE(head(h'), gt) + 0.1·MSE(h', h)` |
| 条件信号 | `softmax(logits)·head.weight` → 预测的下一个 token embedding |
| 置信度头部 | `Linear(768→128→1)→Sigmoid`, 标签 = `logprob_after(gt) > logprob_before(gt)` |

## 当前状态

### v3 — Stateful SSM Denoiser（当前，训练中）

- 20000 轨迹 × 16 步 = 320000 个带 GT token 标签的 AR 状态
- Stateful SSM 按轨迹序列处理（BPTT 式训练）
- 训练目标：最大化 GT token 的每步 logprob + KL 正则化
- 详见 `docs/RINA_实验总览.md §12`

### v2 — Stateless MLP + Confidence Head（已诊断）

- MLP denoiser 改善 3/4 prompt
- Confidence head（entropy-based label）不可靠——Romeo 误拦截、Capital of France 误放行
- 诊断：entropy ≠ 质量，v3 切换为 GT token logprob 标签

### v1 — MoHE-RWKV 109M（已归档）

Attractor MoE + RWKV-v7 backbone。ppl=4.9，路由已分化。因 attractor 坍缩放弃。

## 快速开始

```bash
# 安装依赖
pip install torch numpy tqdm

# 训练 stateful denoiser（Phase 0 + Phase 1）
python rina/train_ar.py

# 训练 confidence head
python rina/train_conf.py

# 评估
python rina/eval_multi.py
```

## 包结构

```
rina/
  rwkv_v7_demo.py          ← RWKV-v7 backbone（已 patch kernel + return_h）
  train_ar.py              ← Stateful denoiser 训练（Phase 0 + Phase 1）
  train_conf.py            ← Confidence head 训练
  eval_multi.py            ← 多 prompt 对比
  __init__.py              ← RWKV, RWKV_TOKENIZER, StatefulDenoiser, run_model

kernels/
  wkv7_fp32.cu / .cpp      ← CUDA kernel（WindBackstepping, fp32 + backward）

checkpoints/
  mohe_fw_rwkv_1b.npy      ← 训练数据（3.7GB）
  rwkv_vocab_v20230424.txt
  ar_trajs.pt              ← v3 轨迹数据
  dn_stateful_final.pt     ← v3 训练好的 denoiser

docs/
  RINA实验日志.md           ← 完整实验记录（8200+ 行）
  RINA_实验总览.md           ← 整理版实验概览
  DLM_survey.md             ← Diffusion LM 文献调研

archive/                   ← 前代归档（CANN, MoHE）
```

## 硬件

ROG Zephyrus M16 2022 — RTX 3070 Ti Laptop (8 GB)

## 参考

- `docs/RINA实验日志.md` — 完整实验记录
- `docs/RINA_实验总览.md` — 整理版实验概览
- `docs/DLM_survey.md` — Diffusion LM 文献调研

## 联系方式

rapidsound@163.com / mikotomisaka477@gmail.com
