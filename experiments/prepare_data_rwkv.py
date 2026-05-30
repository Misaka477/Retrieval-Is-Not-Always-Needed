"""Tokenize FW + SC + Math + Chinese with RWKV tokenizer, multi-threaded."""
import sys, os, io, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

VOCAB_FILE = os.path.join(CKPT_DIR, "rwkv_vocab_v20230424.txt")
SEQ = 128
NUM_WORKERS = 8

# 50% FW + 20% SC + 15% Math + 15% Chinese = 500M tokens
TARGETS = [
    ("FW", "HuggingFaceFW/fineweb", "sample-10BT", "train", 0.50, 250_000_000),
    ("SC", "bigcode/starcoderdata", None, "train", 0.20, 100_000_000),
    ("Math", "open-web-math/open-web-math", None, "train", 0.15, 75_000_000),
    ("Chinese", "Skywork/SkyPile-150B", None, "train", 0.15, 75_000_000),
]

from rina.rwkv_tokenizer import TRIE_TOKENIZER
print("Loading RWKV trie tokenizer...")
_tokenizer = TRIE_TOKENIZER(VOCAB_FILE)
print(f"  Vocab: {len(_tokenizer.idx2token)} tokens")


def np_save_chunked(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    data = buf.getvalue()
    CHUNK = 64 * 1024 * 1024
    with open(path, 'wb') as f:
        for i in range(0, len(data), CHUNK):
            f.write(data[i:i + CHUNK])


def _tokenize_batch(texts):
    return [_tokenizer.encode(t) for t in texts]


def load_or_tokenize(name, src, cfg, split, max_tokens, cache_name):
    cache_path = os.path.join(CKPT_DIR, cache_name)
    if os.path.exists(cache_path):
        ids = np.load(cache_path, mmap_mode='r')
        print(f"  {name}: {len(ids):,} tokens (cached)")
        return torch.from_numpy(ids)
    print(f"  Loading {name} ({max_tokens:,} tokens)...")
    ds = load_dataset(src, cfg, split=split, streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    arr = np.empty(max_tokens, dtype=np.int32)
    pos = 0
    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    batch = []
    it = iter(ds)
    pbar = tqdm(desc=f"  Tokenizing {name}")
    with pool:
        futures = []
        while pos < max_tokens:
            try:
                s = next(it)
            except StopIteration:
                break
            text = s.get("text", s.get("content", ""))
            if len(text) < 200:
                continue
            batch.append(text)
            if len(batch) >= 500:
                futures.append(pool.submit(_tokenize_batch, batch))
                batch = []
            if len(futures) >= NUM_WORKERS * 2:
                for f in as_completed(futures):
                    for tids in f.result():
                        tids = tids[:SEQ * 100]
                        if len(tids) >= SEQ:
                            take = min(len(tids), max_tokens - pos)
                            arr[pos:pos+take] = tids[:take]
                            pos += take
                            pbar.update(1)
                            if pos >= max_tokens:
                                break
                futures = []
                if pos >= max_tokens:
                    break
        if batch:
            futures.append(pool.submit(_tokenize_batch, batch))
        for f in as_completed(futures):
            for tids in f.result():
                tids = tids[:SEQ * 100]
                if len(tids) >= SEQ:
                    take = min(len(tids), max_tokens - pos)
                    arr[pos:pos+take] = tids[:take]
                    pos += take
                    pbar.update(1)
                    if pos >= max_tokens:
                        break
    pbar.close()
    ids = arr[:pos]
    np_save_chunked(cache_path, ids)
    print(f"  {name}: {len(ids):,} tokens (saved)")
    return torch.from_numpy(ids)


all_ids = []
for name, src, cfg, split, ratio, max_tok in TARGETS:
    cache = f"mohe_raw_{name.lower()}_rwkv.npy"
    ids = load_or_tokenize(name, src, cfg, split, max_tok, cache)
    ids = ids[:max_tok]
    all_ids.append(ids)
    print(f"  {name}: taking {len(ids):,} tokens ({ratio*100:.0f}%)")

ids = torch.cat(all_ids)
perm = torch.randperm(len(ids))
ids = ids[perm].numpy()

out_path = os.path.join(CKPT_DIR, "mohe_fw_rwkv.npy")
np_save_chunked(out_path, ids)
print(f"\nSaved combined dataset to {out_path}")
print(f"  Total tokens: {len(ids):,}")
print(f"  Composition: " + " + ".join(f"{t[0]} {int(t[4]*100)}%" for t in TARGETS))
print(f"  Max token ID: {ids.max()}, Min: {ids.min()}")
