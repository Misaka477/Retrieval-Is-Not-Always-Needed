"""下载 DCLM 数据，GPT-2 tokenize，保存为 .npy"""
import os, math
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT_NPY = 'data/dclm_pretrain_llama.npy'
TARGET_TOKENS = 500_000_000  # 500M
os.makedirs('data', exist_ok=True)

print('Loading tokenizer...')
tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token

print('Downloading DCLM (streaming)...')
ds = load_dataset('mlfoundations/dclm-baseline-1.0', split='train', streaming=True)

all_tokens = []
for i, example in enumerate(ds):
    text = example.get('text', example.get('content', ''))
    if not text or len(text.strip()) < 500:
        continue
    ids = tok.encode(text, max_length=10240, truncation=True)
    all_tokens.extend(ids)
    if len(all_tokens) >= TARGET_TOKENS:
        break
    if i % 5000 == 0:
        print(f'  {i} docs, {len(all_tokens)/1e6:.1f}M tokens', flush=True)

data = np.array(all_tokens[:TARGET_TOKENS], dtype=np.int32)
np.save(OUT_NPY, data)
print(f'Saved {len(data)} tokens to {OUT_NPY}')
