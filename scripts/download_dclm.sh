#!/bin/bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_TOKEN=''
cd /home/aquama/Development/RINA_Project
/home/aquama/miniconda3/envs/natalia/bin/python3 -u -c "
import os, sys, numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from huggingface_hub import login

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

login(token='', add_to_git_credential=True)
print('Logged in')

tok = GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer')
tok.pad_token = tok.eos_token

print('Loading DCLM...')
ds = load_dataset('mlfoundations/dclm-baseline-1.0', split='train', streaming=True)

tokens = []
for i, ex in enumerate(ds):
    t = ex['text']
    if len(t.strip()) < 500: continue
    ids = tok.encode(t, max_length=10240, truncation=True)
    tokens.extend(ids)
    if len(tokens) >= 100_000_000: break
    if i % 5000 == 0:
        print(f'{i} docs, {len(tokens)/1e6:.1f}M tokens')
        sys.stdout.flush()

os.makedirs('data', exist_ok=True)
data = np.array(tokens[:100_000_000], dtype=np.int32)
np.save('data/english_pretrain.npy', data)
print(f'Done: {len(data)} tokens')
" >> /tmp/dclm.log 2>&1
echo "DOWNLOAD FINISHED" >> /tmp/dclm.log
