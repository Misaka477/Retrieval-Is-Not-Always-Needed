"""合并新扩展数据为训练用 npy（直接拼接，训练时随机位置 = 天然 shuffle）"""
import os, sys, numpy as np

TARGET = 'data/mix_pretrain_llama_v2.npy'

sources = [
    'data/dclm_pretrain_llama_2000m.npy',
    'data/starcode_pretrain_llama_800m.npy',
    'data/math_pretrain_llama_600m.npy',
    'data/chinese_pretrain_llama_600m.npy',
]

parts = []
total = 0
for src in sources:
    if not os.path.exists(src):
        print(f'跳过: {src} 不存在'); continue
    data = np.load(src, mmap_mode='r')
    parts.append(data)
    total += len(data)
    print(f'{src}: {len(data)/1e6:.0f}M tokens')

full = np.concatenate(parts)
np.save(TARGET, full)
print(f'Done: {len(full)/1e6:.0f}M tokens → {TARGET}')
