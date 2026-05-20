# RINA 开发纪律

- [ ] 实验结果（无论成败）必须写入 `docs/RINA实验日志.md`，含时间戳、配置、数据、结论
- [ ] 每条试错路线必须记录失败原因，不可只记录成功的
- [ ] 新架构想法先在小规模 (dm=256) 验证，确认有效再上 15M
- [ ] 每次 commit 前确认训练/测试脚本能跑通（至少 import 不出错）
- [ ] 参数改动后必须在日志中注明 `CKPT_NAME`，不覆盖旧版 checkpoint
- [ ] 所有对比实验必须控制单一变量，并在日志中列出
- [ ] 论文级结论需要 dm=256 + dm=768 两个尺度交叉验证
- [ ] 所有训练/测试脚本**必须**带 `tqdm` 进度条，不得让运行者盲猜进度
- [ ] 所有训练/测试脚本**必须**输出 `.log` 文件到 `logs/` 目录，含完整的 ppl / att / loss / lr 轨迹
- [ ] 新脚本写完先在小规模上跑通（至少 import + 一步 forward 不崩），再交付
- [ ] **import 顺序规则（防静默退出）：** `os.environ["HF_..."]` → `from tokenizers/datasets import ...` → `import torch` → `torch.manual_seed()`。torch 必须在 datasets 之后导入，否则 CUDA + multiprocessing fork 僵死审查