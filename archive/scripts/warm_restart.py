"""Warm-restart: ep11-13 from checkpoint, LR reset to 3e-4 (matching V1 ep8-10 restarts)."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time
from tqdm import tqdm
torch.manual_seed(42)
from modules.temporal_snn_cell import TemporalSNNModel

device = "cuda"
CKPT_NAME = "cann_snn15m_v2"
CKPT_DIR = "checkpoints"

print("Loading checkpoint...", flush=True)
RESUME_PATH = os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt")
start_ep = 11
if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    start_ep = ckpt["ep"]
    print(f"Resuming from ep {start_ep}", flush=True)
else:
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep10.pt"), map_location=device, weights_only=False)

DM = 840; NP = 4096; V = 4096
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=2,
                          error_threshold=1.0, hebbian_lr=0.01, inhibition_threshold=0.8).to(device)
model.load_state_dict(ckpt["model"])

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)

# Reload data with SAME tokenizer
print("Loading data...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw_text = [t for t in ds["text"] if len(t) > 100][:200000]
tok = Tokenizer.from_file(os.path.join(CKPT_DIR, CKPT_NAME))
V = tok.get_vocab_size()
print(f"  vocab: {V}", flush=True)
ids_list = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw_text if len(tok.encode(t).ids) > 64]
ids = torch.cat(ids_list)
BS, SEQ, SS = 8, 64, 8
num_ba = len(ids) // (BS * SEQ) // SS
print(f"  batches/epoch: {num_ba}", flush=True)

# Train ep11-13
for ep in range(start_ep, 14):
    model.train(); tl = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(num_ba), desc=f"ep {ep}/13", leave=False)
    for bi in pbar:
        s = perm[(bi * SS) % len(perm)]
        x = ids[s:s + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); tl += loss.item()
    lr_scheduler.step()
    ppl = torch.exp(torch.tensor(tl / num_ba)).item()
    att = model.get_att_rate()
    msg = f"ep {ep:2d}: loss={tl/num_ba:.4f} ppl={ppl:.1f} att={att*100:.0f}% lr={opt.param_groups[0]['lr']:.1e}"
    print(msg, flush=True)
    with open(os.path.join(CKPT_DIR, f"{CKPT_NAME}_warmup.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(), "ep": ep, "ppl": ppl,
    }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(), "ep": ep, "ppl": ppl,
    }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt"))
    tok.save(os.path.join(CKPT_DIR, CKPT_NAME))

torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_final.pt"))
print(f"\nDone. ep11-13 warm-restart complete.")
