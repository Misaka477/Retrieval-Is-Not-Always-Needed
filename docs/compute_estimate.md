"""
Compute cost estimate for RINA 15M scaling experiments.
"""
print("=" * 60)
print("RINA 15M 算力与时间估算")
print("=" * 60)

# Base assumptions
tokens_per_experiment = 200_000_000  # 200M tokens per dataset
seq_params = [(64, 8), (128, 4), (256, 2)]  # (seq, bs) pairs

# Hardware profiles
hw = {
    "3070 Ti (当前)": {"it_s": 3.0,  "cost_h": 0.0,   "free": True},
    "Kaggle T4 free": {"it_s": 3.0,  "cost_h": 0.0,   "free": True, "limit_h": 30},
    "Colab T4 free":  {"it_s": 3.0,  "cost_h": 0.0,   "free": True, "limit_h": 12},
    "AutoDL 4090":    {"it_s": 8.0,  "cost_h": 3.5,   "free": False},
    "RunPod 4090":    {"it_s": 8.0,  "cost_h": 0.4,   "free": False},
    "Vast.ai 4090":   {"it_s": 8.0,  "cost_h": 0.35,  "free": False},
    "Vast A100-80G":  {"it_s": 20.0, "cost_h": 1.5,   "free": False},
}

for hw_name, spec in hw.items():
    print(f"\n── {hw_name} ──")
    total_h = 0
    total_cost = 0
    for seq, bs in seq_params:
        tok_per_step = bs * seq
        speed = spec["it_s"]
        h_per_epoch = 200_000_000 / (tok_per_step * speed) / 3600
        total_h += h_per_epoch
    
    cost = total_h * spec.get("cost_h", 0)
    free_str = ""
    if spec.get("free") and spec.get("limit_h"):
        sessions = max(1, int(total_h / spec["limit_h"]) + 1)
        free_str = f"  ⚠️ 需 {sessions} 次会话（限 {spec['limit_h']}h/次）"
    
    print(f"  总时间: {total_h:.0f}h ({total_h/24:.1f} 天)")
    if cost > 0:
        print(f"  总费用: ¥{cost*7.2:.0f} / ${cost:.0f}")
    print(f"  {'✅ 免费' if spec['free'] else '💰 付费'}{free_str}")

print("\n" + "=" * 60)
print("单次实验（200M tokens，含 seq=64+128+256 渐进）")
print("=" * 60)

print("\n备注:")
print("- 渐进 seq=64→128→256 每级训 1 epoch")
print("- 一次完整从零训 ≈ 3 个实验 (FW-Edu 200M + FW 200M + Code 200M)")
print("- 3070 Ti 完成 3 次实验 ≈ 3 × 26h = 78h ≈ 3.2 天连续")
