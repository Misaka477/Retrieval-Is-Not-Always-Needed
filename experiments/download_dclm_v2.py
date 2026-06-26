"""下载 DCLM，按块 tokenize 落盘，内存可控"""
import os, sys, time, gc
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

n_m = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TARGET = n_m * 1_000_000
CHUNK = 100_000_000
OUT_NPY = f'data/dclm_pretrain_llama_{n_m}m.npy'

EXISTING = 0
if os.path.exists('data/dclm_pretrain_llama.npy'):
    EXISTING = np.load('data/dclm_pretrain_llama.npy', mmap_mode='r').shape[0]
    print(f'已有: {EXISTING/1e6:.0f}M')

tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token
print(f'目标: {n_m}M (跳过 ~{EXISTING/1e6:.0f}M, 每 {CHUNK/1e6:.0f}M 落盘)')

ds = load_dataset('mlfoundations/dclm-baseline-1.0', split='train', streaming=True)

skip_est, collected, tmp_idx, buf = 0, 0, 0, []
t0 = time.time()
batch, BS = [], 5000

for i, ex in enumerate(ds):
    text = ex.get('text', '') or ''
    if len(text.strip()) < 500:
        continue

    est = len(text) // 4
    if skip_est < EXISTING:
        skip_est += est
        if i % 10000 == 0:
            print(f'  skip {i}... {skip_est/1e6:.0f}/{EXISTING/1e6:.0f}M', flush=True)
        continue

    batch.append(text)
    if len(batch) < BS:
        continue

    encoded = tok(batch, truncation=True, max_length=10240)
    batch = []
    for ids in encoded['input_ids']:
        buf.extend(ids)
        collected += len(ids)

    if collected - tmp_idx * CHUNK >= CHUNK:
        tmp_name = f'data/_dclm_tmp_{tmp_idx}.npy'
        np.save(tmp_name, np.array(buf, dtype=np.int32))
        tmp_idx += 1
        buf = []
        speed = collected / (time.time() - t0) / 1e6 * 60
        print(f'  docs={i} col={collected/1e6:.0f}M ({collected/TARGET*100:.0f}%) {speed:.0f}M/min', flush=True)

    if collected >= TARGET:
        break
    del encoded; gc.collect()

if buf:
    tmp_name = f'data/_dclm_tmp_{tmp_idx}.npy'
    np.save(tmp_name, np.array(buf, dtype=np.int32))
    tmp_idx += 1

print(f'Merging {tmp_idx} chunks...')
parts = [np.load(f'data/_dclm_tmp_{i}.npy', mmap_mode='r') for i in range(tmp_idx)]
full = np.concatenate(parts)[:TARGET]
np.save(OUT_NPY, full)
for i in range(tmp_idx):
    os.remove(f'data/_dclm_tmp_{i}.npy')
print(f'Done: {len(full)} -> {OUT_NPY} in {(time.time()-t0)/60:.1f}m')
