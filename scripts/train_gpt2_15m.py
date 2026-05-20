"""GPT-2 15M baseline on WikiText-103 — same data as CANN-SSM 15M."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)

SEQ, BS, AE = 64, 8, 2
EPOCHS, LR = 10, 3e-4
N_SEGMENTS = 200000
SUBSAMPLE = 8
CKPT, CKPT_NAME = "checkpoints", "gpt2_15m_wt103"
CKPT_RESUME = f"{CKPT}/{CKPT_NAME}_resume.pt"

# ── Data (same as CANN-SSM) ──
print("Loading WikiText-103...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw_text = [t for t in ds["text"] if len(t) > 100][:N_SEGMENTS]
print(f"  segments: {len(raw_text)}", flush=True)

print("Training BPE...", flush=True)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trn = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>"])
tok.train_from_iterator(raw_text[:30000], trn)
tok.add_special_tokens(["<pad>"])
V = tok.get_vocab_size()
print(f"  vocab: {V}", flush=True)

print("Tokenizing...", flush=True)
ids_list = []
for t in raw_text:
    enc = tok.encode(t).ids
    if len(enc) > SEQ:
        ids_list.append(torch.tensor(enc, dtype=torch.long))
ids = torch.cat(ids_list)
print(f"  tokens: {len(ids):,}", flush=True)

# ── Model (14.8M GPT-2) ──
cfg = GPT2Config(vocab_size=V, n_embd=416, n_layer=6, n_head=8, n_positions=SEQ)
model = GPT2LMHeadModel(cfg).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
os.makedirs(CKPT, exist_ok=True)
tok.save(f"{CKPT}/{CKPT_NAME}-vocab.json")

BATCHES = len(ids) // (BS * SEQ) // SUBSAMPLE
print(f"  batches/epoch: {BATCHES}", flush=True)

start_ep = 0
t_start = time.time()
if os.path.exists(CKPT_RESUME):
    ckpt = torch.load(CKPT_RESUME, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_ep = ckpt["epoch"] + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - start_ep)
    opt.param_groups[0]['lr'] = LR
    print(f"  Resumed from epoch {start_ep} (LR reset to {LR})", flush=True)

for ep in range(start_ep, EPOCHS):
    model.train()
    tot, cnt = 0.0, 0
    pbar = tqdm(range(BATCHES), desc=f"ep {ep+1}/{EPOCHS}")
    for _ in pbar:
        idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
        x = ids[idx[:, None] + torch.arange(SEQ)].to(device)
        y = ids[idx[:, None] + torch.arange(1, SEQ + 1)].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x).logits
            loss = F.cross_entropy(out.view(-1, V), y.reshape(-1))

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tot += loss.item(); cnt += 1
        lr = opt.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{tot/cnt:.3f}", lr=f"{lr:.1e}")

    scheduler.step()
    ppl = torch.exp(torch.tensor(tot/cnt)).item()
    print(f"ep {ep+1:2d}: loss={tot/cnt:.3f} ppl={ppl:.1f} lr={opt.param_groups[0]['lr']:.1e} ({(time.time()-t_start)/60:.1f}min)", flush=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    epoch=ep), CKPT_RESUME)

torch.save(model.state_dict(), f"{CKPT}/{CKPT_NAME}_final.pt")
print(f"Done. {(time.time()-t_start)/60:.1f}min", flush=True)
