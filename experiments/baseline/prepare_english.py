"""下载 DCLM 纯英文数据，GPT-2 tokenize，保存为 .npy"""
import os, json
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer

TOKEN_DIR = 'checkpoints/gpt2_tokenizer'
OUT_PATH = 'data/english_pretrain.npy'
TARGET_TOKENS = 100_000_000  # 100M token

print('Loading GPT-2 tokenizer...')
tok = GPT2Tokenizer.from_pretrained(TOKEN_DIR)
tok.pad_token = tok.eos_token

print('Loading DCLM baseline (streaming)...')
ds = load_dataset('mlfoundations/dclm-baseline-1.0', split='train', streaming=True)

all_tokens = []
total_chars = 0
for i, example in enumerate(ds):
    text = example['text']
    if len(text.strip()) < 500: continue
    ids = tok.encode(text, max_length=10240, truncation=True)
    all_tokens.extend(ids)
    total_chars += len(text)
    if len(all_tokens) >= TARGET_TOKENS:
        break
    if i % 10000 == 0:
        print(f'  {i} docs, {len(all_tokens)/1e6:.1f}M tokens')

data = np.array(all_tokens[:TARGET_TOKENS], dtype=np.int32)
np.save(OUT_PATH, data)
print(f'Saved {len(data)} tokens to {OUT_PATH}')
print(f'Min token: {data.min()}, Max token: {data.max()}')
