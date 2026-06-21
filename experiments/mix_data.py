"""混合 DCLM + StarCode + Math + Chinese 为单一训练数据集"""
import os, numpy as np

DATA_DIR = 'data'
FILES = {
    'dclm':    f'{DATA_DIR}/dclm_pretrain_llama.npy',
    'starcode': f'{DATA_DIR}/starcode_pretrain_llama.npy',
    'math':    f'{DATA_DIR}/math_pretrain_llama.npy',
    'chinese': f'{DATA_DIR}/chinese_pretrain_llama.npy',
}
OUT_NPY = f'{DATA_DIR}/mix_pretrain_llama.npy'

# 加载各数据集
sizes = {}
segments = []
for name, path in FILES.items():
    if not os.path.exists(path):
        print(f'⚠️ {name}: 文件不存在')
        continue
    data = np.load(path, mmap_mode='r')
    n = data.shape[0] - (data.shape[0] % 512)  # 对齐到 seq=512
    segments.append((name, data[:n]))
    sizes[name] = n
    print(f'{name}: {n/1e6:.1f}M tokens')

total = sum(n for _, n in sizes.items())
print(f'总计: {total/1e6:.1f}M tokens')

# 随机截取到目标大小后拼接 + shuffle
np.random.seed(42)

# 按 seq=512 分块后 shuffle（避免模型按顺序看到各数据源）
seq_len = 512
n_seqs = 0
seqs = []
for name, seg in segments:
    n = len(seg) // seq_len
    seqs.append(seg[:n * seq_len].reshape(-1, seq_len))
    n_seqs += n
    print(f'{name}: {n} 个 seq ({n * seq_len / 1e6:.1f}M tokens)')

all_seqs = np.concatenate(seqs, axis=0)  # [total_seqs, 512]
print(f'总 seq 数: {len(all_seqs)}')

# 随机打乱
idx = np.random.permutation(len(all_seqs))
mix = all_seqs[idx].reshape(-1)
print(f'混洗后: {len(mix)/1e6:.1f}M tokens')
np.save(OUT_NPY, mix)
print(f'保存到 {OUT_NPY}')
print(f'大小: {len(mix)/1e6:.1f}M tokens')
