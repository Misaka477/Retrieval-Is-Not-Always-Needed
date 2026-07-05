"""Download pure Python pretraining data from the-stack-smol. Filter by seq <= 512."""
import os, numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data/python_pretrain.npy')
MAX_SEQ = 512
TARGET = 60_000_000

def main():
    tok = AutoTokenizer.from_pretrained(
        os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
        local_files_only=True)

    ds = load_dataset('bigcode/the-stack-smol', split='train', streaming=True)
    tokens = []
    files = 0
    for ex in ds:
        if ex['lang'] != 'Python': continue
        ids = tok.encode(ex['content'])
        if len(ids) < 5: continue
        # Sliding window: split long files into MAX_SEQ+1 chunks
        for start in range(0, len(ids), MAX_SEQ):
            chunk = ids[start:start+MAX_SEQ+1]
            if len(chunk) < 50: break  # last chunk too short, discard
            tokens.extend(chunk[:MAX_SEQ+1])
            files += 1
        if len(tokens) >= TARGET:
            break
        if files % 10000 == 0 and files > 0:
            print(f'  {files} chunks, {len(tokens)/1e6:.1f}M tokens')

    total = len(tokens) - (len(tokens) % MAX_SEQ)
    arr = np.array(tokens[:total], dtype=np.int32)
    np.save(OUT, arr)
    print(f'Saved: {len(arr)/1e6:.1f}M tokens from {files} Python files')

if __name__ == '__main__':
    main()
