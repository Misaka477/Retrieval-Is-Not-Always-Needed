"""下载纯英文数据，GPT-2 tokenize，保存为 .npy + NanoGPT .bin"""
import os, math, pickle
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer

TOK_DIR = 'checkpoints/gpt2_tokenizer'
OUT_NPY = 'data/english_pretrain.npy'
OUT_BIN = 'nanoGPT/data/english/train.bin'
VAL_BIN = 'nanoGPT/data/english/val.bin'
TARGET_TOKENS = 100_000_000
os.makedirs('nanoGPT/data/english', exist_ok=True)

print('Loading tokenizer...')
tok = GPT2Tokenizer.from_pretrained(TOK_DIR)
tok.pad_token = tok.eos_token

print('Downloading FineWeb-Edu (streaming)...')
ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)

all_tokens = []
for i, example in enumerate(ds):
    text = example['text']
    if len(text.strip()) < 500: continue
    ids = tok.encode(text, max_length=10240, truncation=True)
    all_tokens.extend(ids)
    if len(all_tokens) >= TARGET_TOKENS:
        break
    if i % 5000 == 0:
        print(f'  {i} docs, {len(all_tokens)/1e6:.1f}M tokens')

data = np.array(all_tokens[:TARGET_TOKENS], dtype=np.int32)
np.save(OUT_NPY, data)
print(f'Saved {len(data)} tokens to {OUT_NPY}')

# NanoGPT format
train_len = int(len(data) * 0.9)
data[:train_len].astype(np.uint16).tofile(OUT_BIN)
data[train_len:].astype(np.uint16).tofile(VAL_BIN)
with open('nanoGPT/data/english/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': 50257, 'tokenizer': 'gpt2'}, f)
print(f'NanoGPT format: train={train_len}, val={len(data)-train_len}')
