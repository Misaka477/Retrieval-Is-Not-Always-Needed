"""Download and prepare pure Python pretraining data (filtered, seq <= 512)."""
import os, sys, numpy as np
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data/python_raw.npy')

def main(target_tokens=200_000_000):
    tok = AutoTokenizer.from_pretrained(
        os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
        local_files_only=True)
    
    from datasets import load_dataset
    ds = load_dataset('codeparrot/github-code', streaming=True, trust_remote_code=True, split='train')
    
    tokens = []
    count = 0
    for ex in ds:
        if ex['language'] != 'Python': continue
        ids = tok.encode(ex['code'])[:2048]
        if 50 < len(ids) <= 512:
            tokens.extend(ids)
            count += 1
        if len(tokens) >= target_tokens:
            break
        if count % 10000 == 0 and count > 0:
            print(f'  {count} files, {len(tokens)/1e6:.0f}M tokens')
    
    arr = np.array(tokens[:target_tokens - (target_tokens % 512)], dtype=np.int32)
    np.save(OUT, arr)
    print(f'Saved python_raw.npy: {len(arr)/1e6:.0f}M tokens from {count} Python files')

if __name__ == '__main__':
    main()
