"""
SNN v2 with slot-aware training: mix ~10% NIAH samples into WikiText-103.
Goal: eliminate the 22% real-text NIAH ceiling by training slot trust from the start.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)

# ── Config ──
DM, NP, SEQ, BS, AE = 840, 4096, 64, 8, 2
EPOCHS, LR = 13, 3e-4
ERROR_TH = 1.0
HEBB_LR = 0.01
INHIB_TH = 0.8
N_SEGMENTS = 200000
SUBSAMPLE = 8
N_SLOTS = 4096
NIAH_RATIO = 0.10  # 10% of batches are NIAH
CKPT_NAME = "cann_snn15m_v2_slot"

print(f"Config: dm={DM} np={NP} seq={SEQ} bs={BS} ae={AE} th={ERROR_TH}")
print(f"slot={N_SLOTS} niah_ratio={NIAH_RATIO}")

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

def make_niah_batch(bs, seq, vocab_size):
    x = torch.randint(vocab_size, (bs, seq))
    key_id = random.randint(2, vocab_size - 1)
    val_id = random.randint(2, vocab_size - 1)
    while val_id == key_id:
        val_id = random.randint(2, vocab_size - 1)
    kv_pos = random.randint(0, seq - 4)
    x[:, kv_pos] = key_id      # key
    x[:, kv_pos + 1] = val_id   # value
    x[:, -2] = key_id           # query at -2: logits[-1] → loss 参与预测 value
    x[:, -1] = val_id           # target: model predicts value at -1 from slot at -2
    return x, key_id, val_id
ids_list = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw_text if len(tok.encode(t).ids) > SEQ]
ids = torch.cat(ids_list)
print(f"  tokens: {len(ids):,}", flush=True)


# ── Model ──
print("Building Temporal SNN model with slot...", flush=True)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=ERROR_TH,
                          hebbian_lr=HEBB_LR, inhibition_threshold=INHIB_TH,
                           n_slots=N_SLOTS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

# Try torch.compile (PyTorch 2.3+, requires Linux+triton)
try:
    model.cell.forward = torch.compile(model.cell.forward, mode="reduce-overhead",
                                       disable=os.name=="nt")
    if os.name != "nt":
        print(f"  torch.compile: ON (reduce-overhead)", flush=True)
    else:
        print(f"  torch.compile: OFF (Windows, no triton)", flush=True)
except Exception:
    print(f"  torch.compile: OFF (fallback to eager)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

num_batches_per_epoch = (len(ids) - 1) // (BS * SEQ)
train_batches = num_batches_per_epoch // SUBSAMPLE
print(f"  batches/epoch: {train_batches}", flush=True)
niah_batches = max(1, int(train_batches * NIAH_RATIO))
print(f"  niah batches/epoch: {niah_batches} (every {train_batches // niah_batches}th batch)", flush=True)

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
RESUME_PATH = os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt")

start_ep = 1
resume_steps = 0
if os.path.exists(RESUME_PATH):
    print(f"Resuming from {RESUME_PATH}...", flush=True)
    ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    start_ep = ckpt["ep"]
    resume_steps = ckpt.get("train_steps", 0)
    print(f"  resume from ep {start_ep}, step {resume_steps}", flush=True)

tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

for ep in range(start_ep, EPOCHS + 1):
    model.train()
    total_loss = 0.0; lm_loss = 0.0; lm_steps = 0; slot_correct = 0; slot_total = 0
    perm = torch.randperm(len(ids) - BS * SEQ)
    train_batches = num_batches_per_epoch // SUBSAMPLE
    pbar = tqdm(range(train_batches), desc=f"ep {ep}/{EPOCHS}")
    train_steps = resume_steps if ep == start_ep else 0
    skip_until = resume_steps if ep == start_ep else 0
    for bi in pbar:
        train_steps += 1
        if train_steps <= skip_until:
            continue

        if model.n_slots > 0:
            model.slot_table.zero_()

        if bi % (train_batches // niah_batches) == 0 and bi > 0:
            x, key_id, val_id = make_niah_batch(BS, SEQ, V)
            x = x.to(device)
            model.slot_write(key_id, val_id)
            is_niah = True
        else:
            start = perm[(bi * SUBSAMPLE) % len(perm)]
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            is_niah = False

        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            ce_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                x[:, 1:].reshape(-1),
            )
            loss = ce_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        if is_niah:
            slot_total += 1
            if logits[0, -2].argmax().item() == val_id:
                slot_correct += 1
        else:
            lm_loss += loss.item()
            lm_steps += 1
        if train_steps % 500 == 0:
            lm_avg = lm_loss / max(lm_steps, 1)
            ppl = torch.exp(torch.tensor(lm_avg)).item()
            slot_acc = 100 * slot_correct / max(slot_total, 1)
            att = model.get_att_rate()
            pbar.set_postfix(loss=f"{ce_loss.item():.3f}", ppl=f"{ppl:.1f}",
                           slot=f"{slot_acc:.0f}%", att=f"{att*100:.0f}%")
            if train_steps % 2000 == 0:
                torch.save({
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "lr_scheduler": lr_scheduler.state_dict(),
                    "ep": ep, "ppl": ppl, "train_steps": train_steps,
                }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt"))
                tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

    effective_steps = train_steps - resume_steps
    avg_loss = total_loss / effective_steps
    lm_ppl = torch.exp(torch.tensor(lm_loss / max(lm_steps, 1))).item()
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    att_rate = model.get_att_rate()
    lm_ppl = torch.exp(torch.tensor(lm_loss / max(lm_steps, 1))).item()
    slot_acc = 100 * slot_correct / max(slot_total, 1)
    print(f"ep {ep:2d}: ppl={lm_ppl:.1f} slot={slot_acc:.0f}% att={att_rate*100:.0f}% "
          f"lr={opt.param_groups[0]['lr']:.1e}")
    lr_scheduler.step()

    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ep": ep, "ppl": ppl, "lm_ppl": lm_ppl,
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt"))
    tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_final.pt"))
tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))
print(f"\nDone. Final ppl={ppl:.1f}")
print(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")
