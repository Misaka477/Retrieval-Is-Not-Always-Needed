"""下载中文数据（FineWeb-2-zh），GPT-2 tokenize，保存为 .npy"""
import os, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT_NPY = 'data/chinese_pretrain_llama.npy'
TARGET_TOKENS = 150_000_000
os.makedirs('data', exist_ok=True)

tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token

print('Loading FineWeb-2 Chinese...', flush=True)
ds = load_dataset('TiWu-Lab/fineweb-2-zh', split='train', streaming=True)

all_tokens = []
for i, example in enumerate(ds):
    text = example.get('text', example.get('content', ''))
    if not text or len(text.strip()) < 200:
        continue
    ids = tok.encode(text, max_length=8192, truncation=True)
    all_tokens.extend(ids)
    if len(all_tokens) >= TARGET_TOKENS:
        break
    if i % 5000 == 0:
        print(f'  {i} docs, {len(all_tokens)/1e6:.1f}M tokens', flush=True)

data = np.array(all_tokens[:TARGET_TOKENS], dtype=np.int32)
np.save(OUT_NPY, data)
print(f'Saved {len(data)} tokens to {OUT_NPY}', flush=True)
