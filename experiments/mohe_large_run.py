"""
MoHE on FineWeb + StarCoder (200M tokens). 
Depth-of-Thought, winner-take-all Hebbian, GPT-2 50K vocab.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np, io
from rina.mohe import MoHE

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM = 50257, 256
SEQ, BS = 64, 8
LR = 1e-4; EPOCHS = 2
SUBSAMPLE = 8; MAX_TOKENS = 200_000_000
MAX_DEPTH = 1
CONV_THRESH = 0.05; INHIBIT_LR = 0.1
CKPT_DIR = "../checkpoints"; CKPT_NAME = "mohe_large"
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR), exist_ok=True)

# ── Tokenizer ──
print("Loading GPT-2 tokenizer...")
tok = Tokenizer.from_pretrained("gpt2")

# ── Data (FineWeb 80% + StarCoder 20%) ──
CACHE_FW = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, "mohe_fw.npy")
CACHE_SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, "mohe_sc.npy")

def np_save_chunked(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    data = buf.getvalue()
    CHUNK = 64 * 1024 * 1024
    with open(path, 'wb') as f:
        for i in range(0, len(data), CHUNK):
            f.write(data[i:i + CHUNK])

def load_or_tokenize(src, cfg, split, cache, desc, max_tokens=MAX_TOKENS):
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, cache)
    if os.path.exists(cache_path):
        ids = np.load(cache_path, mmap_mode='r')
        ids = torch.from_numpy(ids)
        print(f"  {desc}: {len(ids):,} tokens (cached)")
        return ids
    print(f"  Loading {desc}...")
    ds = load_dataset(src, cfg, split=split, streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    arr = np.empty(max_tokens, dtype=np.int32); pos = 0; batch = []
    for s in tqdm(ds, desc=f"  Tokenizing {desc}"):
        text = s.get("text", s.get("content", ""))
        if len(text) < 200: continue
        batch.append(text)
        if len(batch) >= 500:
            for enc in tok.encode_batch(batch):
                tids = enc.ids[:SEQ * 100]
                if len(tids) >= SEQ:
                    take = min(len(tids), max_tokens - pos)
                    arr[pos:pos+take] = tids[:take]; pos += take
                    if pos >= max_tokens: break
            batch = []
            if pos >= max_tokens: break
    if batch:
        for enc in tok.encode_batch(batch):
            tids = enc.ids[:SEQ * 100]
            if len(tids) >= SEQ:
                take = min(len(tids), max_tokens - pos)
                arr[pos:pos+take] = tids[:take]; pos += take
                if pos >= max_tokens: break
    ids = torch.tensor(arr[:pos], dtype=torch.long)
    np_save_chunked(cache_path, ids.numpy())
    print(f"  {desc}: {len(ids):,} tokens (cached)")
    return ids

ids_fw = load_or_tokenize("HuggingFaceFW/fineweb", "sample-10BT", "train", "mohe_fw.npy", "FineWeb")
ids_sc = load_or_tokenize("bigcode/starcoderdata", None, "train", "mohe_sc.npy", "StarCoder")
ids_math = load_or_tokenize("open-web-math/open-web-math", None, "train", "mohe_math.npy", "OpenWebMath")
# 7:1.5:1.5 (FW : SC : Math)
n_fw = int(MAX_TOKENS * 0.70)
n_sc = int(MAX_TOKENS * 0.15)
n_math = MAX_TOKENS - n_fw - n_sc
ids_fw = ids_fw[:n_fw]
ids_sc = ids_sc[:n_sc]
ids_math = ids_math[:n_math]
# Shuffle interleave
ids = torch.cat([ids_fw, ids_sc, ids_math])
perm = torch.randperm(len(ids))
ids = ids[perm]
nb = (len(ids) - 1) // (BS * SEQ)
print(f"  total: {len(ids):,} tokens, {nb} batches/epoch")

# ── Model ──
model = MoHE(VOCAB, DM, 512, n_experts=4).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / 500))

# ── Resume ──
start_ep = 1
resume_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_resume.pt")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    start_ep = ckpt["ep"]
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed from ep {start_ep}")

# ── Training ──
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}.csv")
print("Training MoHE on FineWeb+StarCoder...")
global_step = 0
for ep in range(start_ep, EPOCHS + 1):
    model.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    its = nb // SUBSAMPLE
    pbar = tqdm(range(its), desc=f"ep {ep}/{EPOCHS}")
    for bi in pbar:
        global_step += 1
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            scheduler.step()
            continue
        loss.backward()
        model.finish_training_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step()
        total_loss += loss.item()
        if bi % 50 == 0:
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"\n[step {bi}] alloc={alloc:.0f}MB reserved={reserved:.0f}MB", flush=True)
        if bi % 200 == 199:
            ppl = torch.exp(torch.tensor(total_loss / (bi + 1))).item()
            lr_now = opt.param_groups[0]['lr']
            pbar.set_postfix(ppl=f"{ppl:.1f}", lr=f"{lr_now:.2e}")
            if not os.path.exists(LOG_PATH):
                with open(LOG_PATH, "w", newline="") as f:
                    f.write("epoch,step,ppl,loss,lr\r\n")
            with open(LOG_PATH, "a", newline="") as f:
                f.write(f"{ep},{global_step},{ppl:.1f},{loss.item():.2f},{lr_now:.2e}\r\n")
        if global_step % 2000 == 0:
            ckpt_mid = {"model": model.state_dict(), "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(), "ep": ep, "ppl": ppl}
            torch.save(ckpt_mid, resume_path)
    ppl = torch.exp(torch.tensor(total_loss / its)).item()
    print(f"  ep {ep}: ppl={ppl:.1f}")
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "ep": ep + 1, "ppl": ppl}
    torch.save(ckpt, resume_path)
    torch.save(ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
