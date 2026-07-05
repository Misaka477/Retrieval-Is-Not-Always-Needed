"""混合数据 v3：砍中文，加大 Code + Math，缩 English，目标 1B tokens"""
import os, sys, numpy as np

OUT = 'data/mix_pretrain_llama_v3.npy'

# 目标比例：Code 40%, Math 25%, English 35%, 中文 0%
sources = [
    ('StarCode', 'data/starcode_pretrain_llama_800m.npy',  400_000_000),   # 40%
    ('Math',     'data/math_pretrain_llama_600m.npy',       250_000_000),   # 25%
    ('DCLM',     'data/dclm_pretrain_llama_2000m.npy',      350_000_000),   # 35%
]

parts = []
total = 0
for name, path, target in sources:
    if not os.path.exists(path):
        print(f'跳过: {path} 不存在'); continue
    data = np.load(path, mmap_mode='r')
    n = min(len(data), target)
    n = n - (n % 512)  # 对齐到 seq=512
    parts.append(data[:n])
    total += n
    print(f'{name}: {n/1e6:.0f}M tokens')

full = np.concatenate(parts)
print(f'总 token: {len(full)/1e6:.0f}M')

# shuffle 分块后打乱
seq_len = 512
all_seqs = full[:len(full) - len(full) % seq_len].reshape(-1, seq_len)
np.random.seed(42)
idx = np.random.permutation(len(all_seqs))
mix = all_seqs[idx].reshape(-1)

np.save(OUT, mix)
print(f'保存: {OUT} ({len(mix)/1e6:.0f}M tokens)')
