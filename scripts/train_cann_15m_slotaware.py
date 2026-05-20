"""15M slot-aware LM training: n_patterns=2048, NIAH mixed every 100 batches."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time, random
from modules.cann_ssm import RINASeqModel, _full_forward
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)

DM, NP, SEQ, BS, AE = 768, 2048, 64, 8, 2
EPOCHS, LR = 10, 3e-4
N_SEGMENTS = 200000; SUBSAMPLE = 8
CKPT, CKPT_NAME = "checkpoints", "cann_15m_slotaware"
NIAH_EVERY = 100  # insert NIAH batch every N batches
GAP = 128

# ── Data ──
print("Loading WikiText-103...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw_text = [t for t in ds["text"] if len(t) > 100][:N_SEGMENTS]
print(f"  segments: {len(raw_text)}", flush=True)

print("Training BPE...", flush=True)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trn = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>"])
tok.train_from_iterator(raw_text[:30000], trn); tok.add_special_tokens(["<pad>"])
V = tok.get_vocab_size()
print(f"  vocab: {V}", flush=True)

print("Tokenizing...", flush=True)
ids_list = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw_text if len(tok.encode(t).ids) > SEQ]
ids = torch.cat(ids_list)
print(f"  tokens: {len(ids):,}", flush=True)

# ── NIAH data generator ──
KEYS = list(range(1, 6)); VALS = list(range(6, 11))
def make_niah_batch():
    seqs, tgts = [], []
    for _ in range(BS):
        k = random.choice(KEYS); v = random.choice(VALS)
        # Snip a random paragraph as filler
        while True:
            p = random.choice(ids_list)
            if len(p) > GAP + 4: break
        start = random.randint(0, len(p) - GAP - 4)
        seq = p[start:start+GAP+4].tolist()
        seq[0] = k; seq[1] = v; seq[-1] = k
        seqs.append(torch.tensor(seq[:256]))  # cap for safety
        tgts.append(v)
    return torch.stack(seqs).to(device), torch.tensor(tgts).to(device)

# ── Model ──
print(f"Model (d={DM}, np={NP})...", flush=True)
model = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=AE).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
os.makedirs(CKPT, exist_ok=True); tok.save(f"{CKPT}/{CKPT_NAME}-vocab.json")

CKPT_RESUME = f"{CKPT}/{CKPT_NAME}_resume.pt"
start_ep = 0; t_start = time.time()
if os.path.exists(CKPT_RESUME):
    ckpt = torch.load(CKPT_RESUME, map_location=device)
    model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"]); start_ep = ckpt["epoch"] + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - start_ep)
    opt.param_groups[0]['lr'] = LR
    print(f"  Resumed from epoch {start_ep}", flush=True)

BATCHES = len(ids) // (BS * SEQ) // SUBSAMPLE
print(f"  batches/epoch: {BATCHES}  niah_every={NIAH_EVERY}", flush=True)

for ep in range(start_ep, EPOCHS):
    model.train(); tot, cnt = 0.0, 0
    pbar = tqdm(range(BATCHES), desc=f"ep {ep+1}/{EPOCHS}")
    for b in pbar:
        # LM batch (regular)
        idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
        x = ids[idx[:, None] + torch.arange(SEQ)].to(device)
        y = ids[idx[:, None] + torch.arange(1, SEQ + 1)].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            out = _full_forward(x, model.embed.weight, model.slot_table,
                model.head.weight, model.head.bias,
                model.state_norm.weight, model.state_norm.bias,
                model.cell.patterns, model.cell.beta_t,
                model.cell.gate_a.weight, model.cell.gate_a.bias,
                model.cell.gate_b.weight, model.cell.gate_b.bias,
                model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                model.cell.proj_in.weight, model.cell.proj_in.bias,
                model.cell.norm.weight, model.cell.norm.bias, AE)
            loss_lm = F.cross_entropy(out.reshape(-1, V), y.reshape(-1))

        opt.zero_grad()
        scaler.scale(loss_lm).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tot += loss_lm.item(); cnt += 1

        # NIAH slot injection every NIAH_EVERY batches
        if b > 0 and b % NIAH_EVERY == 0:
            x_niah, y_niah = make_niah_batch()
            # Write slot BEFORE forward (model sees fresh slot)
            model.slot_table.zero_()
            for i in range(BS):
                model.slot_write(x_niah[i, 0].item(), y_niah[i].item())
            with torch.autocast("cuda", dtype=torch.float16):
                out_n = _full_forward(x_niah, model.embed.weight, model.slot_table,
                    model.head.weight, model.head.bias,
                    model.state_norm.weight, model.state_norm.bias,
                    model.cell.patterns, model.cell.beta_t,
                    model.cell.gate_a.weight, model.cell.gate_a.bias,
                    model.cell.gate_b.weight, model.cell.gate_b.bias,
                    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                    model.cell.proj_in.weight, model.cell.proj_in.bias,
                    model.cell.norm.weight, model.cell.norm.bias, AE)
                loss_niah = F.cross_entropy(out_n[:, -1], y_niah)
            opt.zero_grad()
            scaler.scale(loss_niah).backward()
            scaler.unscale_(opt); scaler.step(opt); scaler.update()
            model.slot_table.zero_()  # reset so LM is clean
            tot += loss_niah.item(); cnt += 1

        lr = opt.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{tot/cnt:.3f}", lr=f"{lr:.1e}")

    scheduler.step()
    ppl = torch.exp(torch.tensor(tot/cnt)).item()
    print(f"ep {ep+1:2d}: loss={tot/cnt:.3f} ppl={ppl:.1f} lr={opt.param_groups[0]['lr']:.1e} ({(time.time()-t_start)/60:.1f}min)", flush=True)
    if (ep+1) % 5 == 0:
        torch.save(model.state_dict(), f"{CKPT}/{CKPT_NAME}_ep{ep+1}.pt")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=ep), CKPT_RESUME)

torch.save(model.state_dict(), f"{CKPT}/{CKPT_NAME}_final.pt")
print(f"Done. {(time.time()-t_start)/60:.1f}min", flush=True)
