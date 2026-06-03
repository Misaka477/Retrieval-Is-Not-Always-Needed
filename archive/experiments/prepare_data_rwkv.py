"""Tokenize FW + SC + Math + Chinese with RWKV tokenizer, multi-threaded."""
import sys, os, io, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

VOCAB_FILE = os.path.join(CKPT_DIR, "rwkv_vocab_v20230424.txt")
SEQ = 128
NUM_WORKERS = 8

# 1B tokens: DCLM 30% + SC 20% + Math 20% + ArXiv 15% + Chinese 15%
TARGETS = [
    ("DCLM", "mlfoundations/dclm-baseline-1.0", None, "train", 0.30, 300_000_000, "text", None),
    ("SC", "bigcode/starcoderdata", None, "train", 0.20, 200_000_000, "content", None),
    ("Math", "open-web-math/open-web-math", None, "train", 0.20, 200_000_000, "text", None),
    ("ArXiv", "monology/pile-uncopyrighted", None, "train", 0.15, 150_000_000, "text", "ArXiv"),
    ("Chinese", "Skywork/SkyPile-150B", None, "train", 0.15, 150_000_000, "text", None),
]

from rina.rwkv_tokenizer import TRIE_TOKENIZER
print("Loading RWKV trie tokenizer...")
_tokenizer = TRIE_TOKENIZER(VOCAB_FILE)
print(f"  Vocab: {len(_tokenizer.idx2token)} tokens")


def np_save_chunked(path, arr):
    tmp = path.replace('.npy', '_tmp.npy')
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception as e:
        np.save(path, arr)


def _tokenize_batch(texts):
    return [_tokenizer.encode(t) for t in texts]


def load_or_tokenize(name, src, cfg, split, max_tokens, cache_name, field="text", pile_filter=None):
    cache_path = os.path.join(CKPT_DIR, cache_name)
    if os.path.exists(cache_path):
        ids = np.load(cache_path, mmap_mode='r')
        if len(ids) >= max_tokens:
            print(f"  {name}: {len(ids):,} tokens (cached)")
            return torch.from_numpy(ids)
        print(f"  {name}: partial cache ({len(ids):,}/{max_tokens:,}), resuming...")
    print(f"  Loading {name} ({max_tokens:,} tokens)...")
    ds = load_dataset(src, cfg, split=split, streaming=True)
    if pile_filter is None:
        ds = ds.shuffle(seed=42, buffer_size=10000)
    arr = np.empty(max_tokens, dtype=np.int32)
    pos = 0; last_save = 0
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode='r')
        arr[:len(cached)] = cached
        pos = len(cached)
        last_save = pos
        print(f"    {pos:,} tokens loaded from partial cache")
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
            if pile_filter and s.get("meta", {}).get("pile_set_name") != pile_filter:
                continue
            text = s.get(field, "")
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
            if pos - last_save >= 10_000_000:
                np_save_chunked(cache_path, arr[:pos])
                last_save = pos
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
total_tokens = 0
for idx, (name, src, cfg, split, ratio, max_tok, field, pile_filter) in enumerate(TARGETS):
    cache = f"mohe_raw_{name.lower()}_rwkv.npy"
    ids = load_or_tokenize(name, src, cfg, split, max_tok, cache, field, pile_filter)
    ids = ids[:max_tok]
    all_ids.append(ids.numpy())
    total_tokens += len(ids)
    print(f"  {name}: taking {len(ids):,} tokens ({ratio*100:.0f}%)")
    partial = np.concatenate(all_ids)
    np.random.shuffle(partial)
    np_save_chunked(os.path.join(CKPT_DIR, "mohe_fw_rwkv_1b.npy"), partial)
    del partial
    print(f"  Partial combined saved ({total_tokens:,} tokens so far)")

print(f"\nAll sources complete: {total_tokens:,} tokens")
print(f"  Composition: " + " + ".join(f"{t[0]} {int(t[4]*100)}%" for t in TARGETS))
