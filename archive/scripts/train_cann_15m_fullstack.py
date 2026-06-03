"""15M full-stack: n_patterns=2048 + slot-aware + predictive coding mixed training."""
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
CKPT, CKPT_NAME = "checkpoints", "cann_15m_fullstack"
NIAH_EVERY, PRED_LAMBDA = 100, 0.05
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

# NIAH: random rare tokens as keys/values (2000-3000 range, mid-freq, never in benchmark)
def random_key(): return random.randint(2000, 3000)
def random_val(): return random.randint(2000, 3000)
def make_niah_batch():
    seqs, tgts = [], []
    for _ in range(BS):
        k = random_key(); v = random_val()
        while True:
            p = random.choice(ids_list)
            if len(p) > GAP + 4: break
        start = random.randint(0, len(p) - GAP - 4)
        seq = p[start:start+GAP+4].tolist()
        ins = random.randint(2, GAP)  # random insert position (not at 0)
        seq[ins] = k; seq[ins+1] = v; seq[-1] = k
        seqs.append(torch.tensor(seq[:256]))
        tgts.append(v)
    return torch.stack(seqs).to(device), torch.tensor(tgts).to(device)

# ── Model ──
from modules.cann_ssm import RINASeqModel
print(f"Model (d={DM}, np={NP})...", flush=True)
model = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=AE).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

# Predictive coding head: h_t -> predict h_{t+1}
pred_head = torch.nn.Linear(DM, DM).to(device)

opt = torch.optim.AdamW(list(model.parameters()) + list(pred_head.parameters()), lr=LR)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
os.makedirs(CKPT, exist_ok=True); tok.save(f"{CKPT}/{CKPT_NAME}-vocab.json")

CKPT_RESUME = f"{CKPT}/{CKPT_NAME}_resume.pt"
start_ep = 0; t_start = time.time()
if os.path.exists(CKPT_RESUME):
    ckpt = torch.load(CKPT_RESUME, map_location=device)
    model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"]); start_ep = ckpt["epoch"] + 1
    pred_head.load_state_dict(ckpt["pred_head"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - start_ep)
    opt.param_groups[0]['lr'] = LR
    print(f"  Resumed from epoch {start_ep}", flush=True)

BATCHES = len(ids) // (BS * SEQ) // SUBSAMPLE
print(f"  batches/epoch: {BATCHES}  niah_every={NIAH_EVERY}  pred_lambda={PRED_LAMBDA}", flush=True)

# ── Sub-cell forward with prediction ──
def extract_h_ssm_for_pred(h, x):
    """Run one cell step (no LN), return h_ssm for prediction coding."""
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ model.cell.gate_a.weight.t() + model.cell.gate_a.bias)
    b = torch.sigmoid(combined @ model.cell.gate_b.weight.t() + model.cell.gate_b.bias)
    h_ssm = a * h + b * (x @ model.cell.proj_in.weight.t() + model.cell.proj_in.bias)
    return h_ssm

for ep in range(start_ep, EPOCHS):
    model.train(); pred_head.train()
    tot, cnt = 0.0, 0
    pbar = tqdm(range(BATCHES), desc=f"ep {ep+1}/{EPOCHS}")
    for b in pbar:
        idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
        x = ids[idx[:, None] + torch.arange(SEQ)].to(device)
        y = ids[idx[:, None] + torch.arange(1, SEQ + 1)].to(device)
        emb = model.embed(x)

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

        # Prediction coding: predict next state from current
        h = torch.zeros(BS, DM, device=device)
        pred_losses = []
        for t in range(SEQ - 1):
            h_ssm_t = extract_h_ssm_for_pred(h, emb[:, t])
            u_pred = pred_head(h_ssm_t)      # predict next h_ssm
            # Get next step's h_ssm (using step t's output h as input h for step t+1)
            h_next = h_ssm_t
            # Simplified: use a lagged MSE
            u_target = extract_h_ssm_for_pred(h_next.detach(), emb[:, t+1]).detach()
            pred_losses.append(F.mse_loss(u_pred.float(), u_target.float()))
            h = h_next
        loss_pred = torch.stack(pred_losses).mean() if pred_losses else torch.tensor(0.0, device=device)
        loss_total = loss_lm + PRED_LAMBDA * loss_pred

        opt.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tot += loss_lm.item(); cnt += 1

        # ── NIAH slot injection every NIAH_EVERY batches ──
        if b > 0 and b % NIAH_EVERY == 0:
            x_niah, y_niah = make_niah_batch()
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
            model.slot_table.zero_()
            tot += loss_niah.item(); cnt += 1

        lr = opt.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{tot/cnt:.3f}", lr=f"{lr:.1e}")

    scheduler.step()
    ppl = torch.exp(torch.tensor(tot/cnt)).item()
    print(f"ep {ep+1:2d}: loss={tot/cnt:.3f} ppl={ppl:.1f} lr={opt.param_groups[0]['lr']:.1e} ({(time.time()-t_start)/60:.1f}min)", flush=True)
    if (ep+1) % 5 == 0:
        torch.save(dict(model=model.state_dict(), pred_head=pred_head.state_dict()), f"{CKPT}/{CKPT_NAME}_ep{ep+1}.pt")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=ep, pred_head=pred_head.state_dict()), CKPT_RESUME)

torch.save(dict(model=model.state_dict(), pred_head=pred_head.state_dict()), f"{CKPT}/{CKPT_NAME}_final.pt")
print(f"Done. {(time.time()-t_start)/60:.1f}min", flush=True)
