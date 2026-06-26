"""下载 StackMathQA，批量 tokenize，跳过已有"""
import os, sys, time, gc, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

n_m = int(sys.argv[1]) if len(sys.argv) > 1 else 600
TARGET = n_m * 1_000_000
OUT_NPY = f'data/math_pretrain_llama_{n_m}m.npy'

EXISTING = 0
if os.path.exists('data/math_pretrain_llama.npy'):
    EXISTING = np.load('data/math_pretrain_llama.npy', mmap_mode='r').shape[0]
    print(f'已有: {EXISTING/1e6:.0f}M')

tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token
print(f'目标: {n_m}M (跳过 ~{EXISTING/1e6:.0f}M)')
AVG_TOK_PER_DOC = 1000
SKIP_DOCS = EXISTING // AVG_TOK_PER_DOC
print(f'跳过 ~{SKIP_DOCS} 个文档')
ds = load_dataset('math-ai/StackMathQA', split='train', streaming=True).skip(SKIP_DOCS)

col, t0 = 0, time.time()
batch, BS, CHUNK, buf, tmp_idx = [], 5000, 100_000_000, [], 0

for i, ex in enumerate(ds):
    q = ex.get('Q', ex.get('question', ''))
    a = ex.get('A', ex.get('answer', ''))
    text = f'Question: {q}\nAnswer: {a}'
    if len(text) < 20: continue
    batch.append(text)
    if len(batch) < BS: continue
    enc = tok(batch, truncation=True, max_length=4096)
    batch = []
    for ids in enc['input_ids']: buf.extend(ids); col += len(ids)
    if col - tmp_idx * CHUNK >= CHUNK:
        np.save(f'data/_math_tmp_{tmp_idx}.npy', np.array(buf, dtype=np.int32))
        tmp_idx += 1; buf = []
        s = col/(time.time()-t0)/1e6*60
        print(f'  sample={i} col={col/1e6:.0f}M ({col/TARGET*100:.0f}%) {s:.0f}M/min', flush=True)
    if col >= TARGET: break
    del enc; gc.collect()

if buf:
    np.save(f'data/_math_tmp_{tmp_idx}.npy', np.array(buf, dtype=np.int32)); tmp_idx += 1
print(f'Merging {tmp_idx} chunks...')
parts = [np.load(f'data/_math_tmp_{i}.npy', mmap_mode='r') for i in range(tmp_idx)]
np.save(OUT_NPY, np.concatenate(parts)[:TARGET])
for i in range(tmp_idx): os.remove(f'data/_math_tmp_{i}.npy')
print(f'Done in {(time.time()-t0)/60:.1f}m')
