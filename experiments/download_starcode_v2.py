"""下载 StarCode，.skip() 跳过已有，按块落盘"""
import os, sys, time, gc
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

n_m = int(sys.argv[1]) if len(sys.argv) > 1 else 800
TARGET = n_m * 1_000_000
CHUNK = 100_000_000
OUT = f'data/starcode_pretrain_llama_{n_m}m.npy'

EXISTING = 0
if os.path.exists('data/starcode_pretrain_llama.npy'):
    EXISTING = np.load('data/starcode_pretrain_llama.npy', mmap_mode='r').shape[0]
    print(f'已有: {EXISTING/1e6:.0f}M tokens')

tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token

# 估算需要跳过的文档数（平均 ~2000 tokens/doc for code）
AVG_TOK_PER_DOC = 2000
SKIP_DOCS = EXISTING // AVG_TOK_PER_DOC
print(f'目标: {n_m}M 新增, 跳过 ~{SKIP_DOCS} 个文档')

ds = load_dataset('bigcode/starcoderdata', split='train', streaming=True).skip(SKIP_DOCS)

tot, col, t0, buf, tmp_idx = 0, 0, time.time(), [], 0
batch, BS = [], 5000

for i, ex in enumerate(ds):
    text = ex.get('content', '') or ''
    if len(text) < 100:
        continue
    batch.append(text)
    if len(batch) < BS:
        continue
    enc = tok(batch, truncation=True, max_length=8192)
    batch = []
    for ids in enc['input_ids']:
        buf.extend(ids); col += len(ids)
    if col - tmp_idx * CHUNK >= CHUNK:
        np.save(f'data/_starcode_tmp_{tmp_idx}.npy', np.array(buf, dtype=np.int32))
        tmp_idx += 1; buf = []
        s = col/(time.time()-t0)/1e6*60
        print(f'  doc={i} col={col/1e6:.0f}M ({col/TARGET*100:.0f}%) {s:.0f}M/min', flush=True)
    if col >= TARGET:
        break
    del enc; gc.collect()

if buf:
    np.save(f'data/_starcode_tmp_{tmp_idx}.npy', np.array(buf, dtype=np.int32))
    tmp_idx += 1

print(f'Merging {tmp_idx} chunks...')
parts = [np.load(f'data/_starcode_tmp_{i}.npy', mmap_mode='r') for i in range(tmp_idx)]
full = np.concatenate(parts)[:TARGET]
np.save(OUT, full)
for i in range(tmp_idx):
    os.remove(f'data/_starcode_tmp_{i}.npy')
print(f'Done: {len(full)} -> {OUT} in {(time.time()-t0)/60:.1f}m')
