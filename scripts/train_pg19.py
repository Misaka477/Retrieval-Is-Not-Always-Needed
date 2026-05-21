"""
PG-19 续训 — 验证 15M 数据天花板是否到来。
加载 slot 训练的 ep12 checkpoint，直接在 PG-19 上训 1-2 epoch。
如果 ppL 继续下降（<34），证明天花板远未到来。
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)

V, DM, NP, AE = 4096, 840, 4096, 2
SEQ, BS = 64, 8
LR = 3e-4; EPOCHS = 3
CKPT_SOURCE = "checkpoints/cann_snn15m_v2_slot_ep12.pt"

print("Loading checkpoint...", flush=True)
sd = torch.load(CKPT_SOURCE, map_location=device, weights_only=False)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
model.load_state_dict(sd["model"], strict=False)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

print("Loading PG-19...", flush=True)
ds = load_dataset("pg19", split="train")
texts = [t for t in ds["text"][:50000] if len(t) > 100]
print(f"  segments: {len(texts)}", flush=True)

print("Tokenizing...", flush=True)
ids_list = []
for t in tqdm(texts, desc="tokenizing"):
    ids = tok.encode(t).ids
    if len(ids) > SEQ:
        ids_list.append(torch.tensor(ids[:SEQ * 1000], dtype=torch.long))
ids = torch.cat(ids_list) if ids_list else torch.zeros(0, dtype=torch.long)
print(f"  tokens: {len(ids):,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
batches_per_epoch = (len(ids) - 1) // (BS * SEQ)
num_batches = batches_per_epoch
print(f"  batches/epoch: {num_batches}", flush=True)

for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(num_batches), desc=f"ep {ep}/{EPOCHS}")
    for bi in pbar:
        start = perm[(bi * 4) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
        if bi % 200 == 0:
            avg_loss = total_loss / max(bi, 1)
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            pbar.set_postfix(loss=f"{avg_loss:.3f}", ppl=f"{ppl:.1f}")

    avg_loss = total_loss / num_batches
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"ep {ep}: ppl={ppl:.1f} on PG-19")
    lr_scheduler.step()
    torch.save({"model": model.state_dict(), "ep": ep, "ppl": ppl},
               os.path.join(CKPT_DIR, f"rina_pg19_ep{ep}.pt"))

print(f"\nDone. PPL on PG-19 after {EPOCHS} epoch: {ppl:.1f}")
if ppl < 34:
    print("↓ Data ceiling NOT reached — scaling to more data is warranted.")
else:
    print("→ PPL plateaued. Further scaling experiments needed to confirm ceiling.")
