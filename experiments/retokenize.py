"""重 tokenize: GPT-2 tokenized data → Llama tokenizer"""
import os, numpy as np
from transformers import GPT2Tokenizer, AutoTokenizer

print('加载 tokenizer...', flush=True)
tok_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
tok_gpt2.pad_token = tok_gpt2.eos_token
tok_llama = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok_llama.pad_token = tok_llama.eos_token

DATA_DIR = 'data'
CHUNK = 100_000  # 每批处理 100K 个 GPT-2 token

for src_name, dst_name in [('mix_pretrain.npy', 'mix_pretrain_llama.npy')]:
    src_path = f'{DATA_DIR}/{src_name}'
    if not os.path.exists(src_path):
        print(f'跳过 {src_name}', flush=True); continue
    
    data = np.load(src_path, mmap_mode='r')
    total = len(data)
    print(f'{src_name}: {total/1e6:.1f}M GPT-2 tokens', flush=True)
    
    # 逐 chunk 解码 → 重编码
    all_ids = []
    for start in range(0, total, CHUNK):
        chunk = data[start:start+CHUNK]
        if len(chunk) < 10:
            all_ids.extend(chunk.tolist())
            continue
        text = tok_gpt2.decode(chunk)
        ids = tok_llama.encode(text, truncation=True, max_length=len(chunk)*2)
        all_ids.extend(ids)
        if start % (CHUNK * 10) == 0:
            print(f'  {(start)/total*100:.0f}%...', flush=True)
    
    result = np.array(all_ids[:total], dtype=np.int32)  # 对齐到原长度
    compression = total / len(result)
    np.save(f'{DATA_DIR}/{dst_name}', result)
    print(f'  → {dst_name}: {len(result)/1e6:.1f}M tokens, 压缩率={compression:.2f}x', flush=True)
