# RINA 项目交接文档 v3 — 2026-06-04

## 项目状态总览

19 天系统实验，覆盖 9 条技术路线（+stateful denoiser）。当前方向：**AR + Stateful Denoiser**。

上一代（v2）诊断：stateless MLP denoiser（Linear(1536→1536)→GELU→Linear(1536→768)+residual）训 MSE 预测 clean state，在 3/4 prompt 上改善，但 conf head 的 entropy-based label 不可靠——Capital of France 的 denoiser 被错误放行（导致 CoT 绕死），Romeo 的 denoiser 被错误拦截（错过了正确答案）。

**当前代（v3）改动：去 MLP + 去 proxy label，换 stateful SSM + per-step GT token logprob 训练。**

## v3 架构

```
官方 12L RWKV-v7 backbone（冻结 + return_h）
  → h [768] → stateful denoiser → h' → head → logits → token

Denoiser（Stateful SSM）:
  SSM 核心（log_A, B, C）：s_t = σ(log_A)·s_{t-1} + B·proj(concat(h_t, cond_t))
  Readout + gated residual：h' = h + sigmoid(gate) · out(C·s_t)
  训练目标: CE(head(h'), gt_token) + β·MSE(h', h), β=0.1, lr=3e-4
  训在 AR 轨迹上（20000 种子 × 16 步 = 320000 带 GT label state）
  状态跨步传递：denoiser 看到过去所有修正历史，可以渐进 refine

Confidence head（重建）:
  Linear(768→128) → ReLU → Linear(128→1) → Sigmoid
  训练目标: BCE(conf, Δlogprob(gt_token) > 0)  ← label 从熵改善换成 GT token logprob 改善
  输入: 单步 raw h_t（不是 trajectory mean）
  推理: h_mix = conf·h' + (1-conf)·h_raw，state 始终推进
```

### 核心区别 vs v2

| 维度 | v2（stateless MLP） | v3（stateful SSM） |
|------|-------------------|-------------------|
| 架构 | 每步独立 MLP | SSM 跨步记忆 |
| 训练信号 | MSE(h_pred, h_clean) 打标自身 | CE(gt_token) 直接质量反馈 |
| Conf label | entropy 下降 | GT token logprob 改善 |
| 数据 | 80000 无序 state | 320000 轨迹组织 state |
| Denoiser 视角 | "这个 state 噪不噪声" | "这个序列修正方向对不对" |

## 当前文件结构

```
rina/
  rwkv_v7_demo.py             ← 官方 backbone（已 patch 内核 + return_h + 绝对路径）
  train_ar.py                 ← v3: Stateful SSM denoiser 训练（Phase 0 + Phase 1）
  train_conf.py               ← v3: Confidence head 训练（per-step GT logprob label）
  eval_multi.py               ← v3: 多 prompt 对比（已改 stateful denoiser）
  official_model.py           ← 旧版官方模型封装（可删）
  rwkv_tokenizer.py / sample.py / __init__.py
kernels/
  wkv7_fp32.cu / .cpp         ← 编译好的 CUDA kernel（WindBackstepping，fp32+backward）
  wkv7_clampw.cu / .cpp       ← 旧 kernel（已归档）
checkpoints/
  mohe_fw_rwkv_1b.npy         ← 训练数据（3.7GB）
  rwkv_vocab_v20230424.txt    ← tokenizer
  ar_trajs.pt                 ← v3: 轨迹数据（20000×16 step, h+cond+gt）
  dn_stateful_final.pt        ← v3: 训好的 stateful denoiser
  dn_conf_stateful.pt         ← v3: 训好的 confidence head（per-step GT logprob）
  dn_ar_final.pt              ← v2: 旧 MLP denoiser（可删/归档）
  ar_states.pt                ← v2: 旧无序 state 数据（可删/归档）
  dn_conf_final.pt            ← v2: 旧 confidence head（可删/归档）
  diff_v2_*.pt                ← 旧实验文件（可删）
rwkv7-g1d-0.1b-20260129-ctx8192.pth  ← 模型权重（0.36GB）
_rwkv_official/               ← RWKV repo（保留，ROSA-1bit 参考）
docs/
  RINA实验日志.md              ← 8000 行完整实验记录
  RINA_实验总览.md              ← 项目总览
  DLM_survey.md                ← Diffusion LM 综述
archive/                      ← 旧模型/脚本/检查点
```

## 实验结果概要

### 正向结果（AR+State Diffusion）

```
Prompt: "Capital of France?"
  AR:      ❌ 答非所问
  AR+Dn:   ❌ CoT 绕死
  AR+Conf: ✅ "The capital of France is Paris"

Prompt: "Eiffel tower is in"
  AR:      ✅ "Paris" 基本事实
  AR+Dn:   ✅ "Paris, France. 1889, Gustave Eiffel" 更详细
  AR+Conf: ❌ 重复退化

Prompt: "Who wrote Romeo and Juliet?"
  AR:      ❌ "Julius Caesar"
  AR+Dn:   ✅ 修正为 Shakespeare（虽绕但答案对了）
  AR+Conf: ❌ Lady Macbeth

Prompt: "Poem about a cat"
  AR:      ✅ 可读
  AR+Dn:   ✅ 最佳（意象最丰富）
  AR+Conf: ✅ 中等可用
```

**v2 结论：denoiser 在某些 prompt 上显著改善，但不是全部。Confidence head 的熵标签不够精准。v3 切换为 stateful SSM + per-step GT logprob 训练，正在等待结果。**

### 有待验证的假设

1. Stateful denoiser 的跨步记忆能否解决 Capital of France 的 CoT 绕死（之前 MLP 每步独立修正 → 方向漂移 → 绕死，stateful 有累积记忆 → 方向一致）
2. GT token logprob 作为 conf label 能否同时修掉 Romeo（之前 entropy label 错误拦截了正确答案）和 Capital of France（之前 entropy label 错误放行了有害修正）

### 关闭的方向

| 方向 | 结论 |
|------|------|
| Attractor MoE | 低秩双向瓶颈导致坍缩 |
| CANN做 LM | 工具选型错误 |
| SSM替代 WKV | 改善记忆但不解决 CE 坍缩 |
| 噪声预测扩散 | denosier 学到输出零 |
| 结构化噪声/top-k条件 | 不如简单方案 |
| 全序列多步扩散 | 推离流形 |

## 核心里程碑

```
2026-05-15: CANN-SSM 出生
2026-05-23: MoHE WKV7 backbone 移植成功
2026-05-25: attractor MoE 验证失败
2026-06-02: 内核修复（双指数 bug）+ CUDA kernel 编译
2026-06-02: 官方 12L backbone 正确加载
2026-06-03: AR+State Diffusion 首次正向结果
2026-06-04: Confidence head 训练
2026-06-04: v3 Stateful SSM denoiser 架构切换
```

## 迁移到 Fedora 前需要做的事

1. **下载安装 Fedora Workstation + Hyprland**
   - `sudo dnf install hyprland` 或在 COPR 装
   - 装 CUDA Toolkit 12.4、torch 2.6

2. **迁移项目文件**
   - 核心文件清单（见上方，约 0.5GB）
   - `.config/kilo/` + `.local/share/kilo/`

3. **Fedora 上需要的软件**
   - flatpak（Spotify、Listen1、MuseScore）
   - `uGet + aria2`（替代 IDM）、`rustdesk`（远程）、`Tailscale`（VPN）
   - Clash Verge Rev 或直接装 VPN 客户端
   - `steam` + `steam-devices` + Proton（VRChat）

## 下一步方向

1. **跑 v3 训练 + 评估** — 先跑 `train_ar.py`（数据采集 + stateful denoiser 训练），再跑 `train_conf.py`（per-step GT logprob conf head），最后 `eval_multi.py` 对比 v2 结果
2. **如果 v3 有效：** 扩展轨迹收集（50000+ 种子 × 32+ 步）、加 depth=3 chain（depth iteration refine）
3. **如果 v3 无效但 mode collapse：** 加两路 expert（事实 vs 创造）+ router（用 reward 端到端训练）
4. **如果 v3 无效且不 collapse：** 走真扩散（DDPM-style time embedding + T 步噪声阶梯），不再寄生在 AR 每一步
5. **读 DLM survey** — 验证和已发表工作的差异，避免重复发明
6. **论文章节** — negative results + AR+State Diffusion 已有一个完整 story

### 训练顺序

```
python rina/train_ar.py        # 20min Phase 0 + ~2h Phase 1 → dn_stateful_final.pt
python rina/train_conf.py      # ~10min → dn_conf_stateful.pt
python rina/eval_multi.py      # ~2min → 看结果对比
```

## 关键经验和教训

1. **训练分布 = 推理分布** — diffuser 成功的必要条件
2. **梯度大小是隐形成本** — reduction='sum'/BSZ 解锁了之前训不动的方向
3. **损失函数设计比架构设计重要** — MSE(head，h_clean) > MSE(noise_pred, noise)
4. **Wolpert 前向/反向模型直觉是有指导意义的** — 体现在 confidence gating 的方向
