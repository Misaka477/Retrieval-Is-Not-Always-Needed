"""CANN-SSM low-rank r=128 training (1.5x speedup, ppl preserved)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time, random
from modules.cann_ssm import RINASeqModel, _full_forward
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)

DM, NP, SEQ, BS, AE, RANK = 880, 4096, 64, 8, 2, 128
EPOCHS, LR = 10, 3e-4
N_SEGMENTS = 200000; SUBSAMPLE = 8
CKPT_NAME = "cann_lowrank_dm880"

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
NIAH_EVERY, GAP = 100, 128
def rk(): return random.randint(2000, 3000)
def rv(): return random.randint(2000, 3000)
def make_niah_batch():
    seqs, tgts, kvs = [], [], []
    for _ in range(BS):
        k, v = rk(), rv()
        while True:
            p = random.choice(ids_list)
            if len(p) > GAP + 4: break
        start = random.randint(0, len(p) - GAP - 4)
        seq = p[start:start+GAP+4].tolist()
        ins = random.randint(2, GAP); seq[ins] = k; seq[ins+1] = v; seq[-1] = k
        seqs.append(torch.tensor(seq[:256])); tgts.append(v); kvs.append((k, v))
    return torch.stack(seqs).to(device), torch.tensor(tgts).to(device), kvs

print(f"Model (low-rank r={RANK}, ae={AE})...", flush=True)
model = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                     attract_every=AE, pattern_rank=RANK).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

pred_head = torch.nn.Linear(DM, DM).to(device)
opt = torch.optim.AdamW(list(model.parameters()) + list(pred_head.parameters()), lr=LR)
PRED_LAMBDA = 0.05
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
os.makedirs("checkpoints", exist_ok=True)

CKPT_RESUME = f"checkpoints/{CKPT_NAME}_resume.pt"
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
print(f"  batches/epoch: {BATCHES}", flush=True)

for ep in range(start_ep, EPOCHS):
    model.train(); tot, cnt = 0.0, 0
    pbar = tqdm(range(BATCHES), desc=f"ep {ep+1}/{EPOCHS}")
    for b in pbar:
        idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
        x = ids[idx[:, None] + torch.arange(SEQ)].to(device)
        y = ids[idx[:, None] + torch.arange(1, SEQ + 1)].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            out = _full_forward(x, model.embed.weight, model.slot_table,
                model.head.weight, model.head.bias,
                model.state_norm.weight, model.state_norm.bias,
                model.cell.effective_patterns, model.cell.beta_t,
                model.cell.gate_a.weight, model.cell.gate_a.bias,
                model.cell.gate_b.weight, model.cell.gate_b.bias,
                model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                model.cell.proj_in.weight, model.cell.proj_in.bias,
                model.cell.norm.weight, model.cell.norm.bias, AE)
            loss = F.cross_entropy(out.reshape(-1, V), y.reshape(-1))

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tot += loss.item(); cnt += 1
        lr = opt.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{tot/cnt:.3f}", lr=f"{lr:.1e}")

        # ── NIAH + prediction coding every NIAH_EVERY batches ──
        if b > 0 and b % NIAH_EVERY == 0:
            x_niah, y_niah, kvs = make_niah_batch()
            model.slot_table.zero_()
            for k, v in kvs: model.slot_write(k, v)

            # Python loop forward for NIAH (allows h_ssm capture)
            h_ssm_seq = []
            h = torch.zeros(BS, DM, device=device)
            emb = model.embed(x_niah)
            for t in range(x_niah.shape[1] - 1):
                a = torch.sigmoid(model.cell.gate_a(torch.cat([h, emb[:, t]], -1)))
                b = torch.sigmoid(model.cell.gate_b(torch.cat([h, emb[:, t]], -1)))
                xp = model.cell.proj_in(emb[:, t])
                h_ssm = a * h + b * xp
                h_ssm_seq.append(h_ssm)
                if t % AE == (AE - 1):
                    scores = (h_ssm @ model.cell.effective_patterns.t()) * model.cell.beta_t[0]
                    attn = torch.softmax(scores, dim=-1)
                    attracted = attn @ model.cell.effective_patterns
                    alpha = torch.sigmoid(model.cell.gate_alpha(torch.cat([h, emb[:, t]], -1)))
                    h_new = h_ssm + alpha * (attracted - h_ssm)
                else:
                    h_new = h_ssm
                h = F.layer_norm(h_new, [DM], model.cell.norm.weight, model.cell.norm.bias, 1e-5)

            # Prediction coding loss on NIAH sequence
            h_ssm_tensor = torch.stack(h_ssm_seq, dim=1)
            if h_ssm_tensor.shape[1] > 1:
                pred = pred_head(h_ssm_tensor[:, :-1])
                target = h_ssm_tensor[:, 1:].detach()
                loss_pred = F.mse_loss(pred.float(), target.float())

            # NIAH recall loss
            i_ext = model.slot_table[x_niah[:, -1]]
            a = torch.sigmoid(model.cell.gate_a(torch.cat([h + i_ext, emb[:, -1]], -1)))
            b = torch.sigmoid(model.cell.gate_b(torch.cat([h + i_ext, emb[:, -1]], -1)))
            xp = model.cell.proj_in(emb[:, -1])
            h_ssm = a * (h + i_ext) + b * xp
            scores = (h_ssm @ model.cell.effective_patterns.t()) * model.cell.beta_t[0]
            attn_n = torch.softmax(scores, dim=-1)
            attracted_n = attn_n @ model.cell.effective_patterns
            alpha = torch.sigmoid(model.cell.gate_alpha(torch.cat([h + i_ext, emb[:, -1]], -1)))
            h_last = h_ssm + alpha * (attracted_n - h_ssm)
            h_last = F.layer_norm(h_last, [DM], model.cell.norm.weight, model.cell.norm.bias, 1e-5)
            logit_last = model.head(model.state_norm(h_last))
            loss_niah = F.cross_entropy(logit_last, y_niah)

            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                total_niah = loss_niah + PRED_LAMBDA * loss_pred
            scaler.scale(total_niah).backward()
            scaler.unscale_(opt); scaler.step(opt); scaler.update()
            model.slot_table.zero_()
            tot += loss_niah.item(); cnt += 1

    scheduler.step()
    ppl = torch.exp(torch.tensor(tot/cnt)).item()
    print(f"ep {ep+1:2d}: loss={tot/cnt:.3f} ppl={ppl:.1f} lr={opt.param_groups[0]['lr']:.1e} ({(time.time()-t_start)/60:.1f}min)", flush=True)
    if (ep+1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints/{CKPT_NAME}_ep{ep+1}.pt")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=ep, pred_head=pred_head.state_dict()), CKPT_RESUME)

torch.save(model.state_dict(), f"checkpoints/{CKPT_NAME}_final.pt")
print(f"Done. {(time.time()-t_start)/60:.1f}min", flush=True)
