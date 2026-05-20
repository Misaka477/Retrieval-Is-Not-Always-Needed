"""
Temporal SNN 15M training — 预测误差门控 + Hebbian 可塑性的完整架构训练。

配置对标 V1 15M (dm=768, np=4096, seq=64, bs=8, WikiText-103)。
对比: V1 ppl=34.5 (全秩) / ppl=40.6 (低秩 r=128)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time, random
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)

# ── 配置 ──
DM, NP, SEQ, BS, AE = 840, 4096, 64, 8, 2
EPOCHS, LR = 13, 3e-4
PRED_LAMBDA = 0.0
ERROR_TH = 1.0
HEBB_LR = 0.01
INHIB_TH = 0.8
N_SEGMENTS = 200000
SUBSAMPLE = 8  # 对齐 V1 配置
CKPT_NAME = "cann_snn15m_v2"

print(f"Config: dm={DM} np={NP} seq={SEQ} bs={BS} ae={AE} th={ERROR_TH} hb_lr={HEBB_LR}")
print(f"Epochs: {EPOCHS}  LR: {LR}")

# ── 数据 ──
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

# ── 模型 ──
print("Building Temporal SNN model...", flush=True)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=ERROR_TH,
                          hebbian_lr=HEBB_LR, inhibition_threshold=INHIB_TH).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# ── 训练 ──
num_batches_per_epoch = (len(ids) - 1) // (BS * SEQ)
print(f"  batches/epoch: {num_batches_per_epoch}", flush=True)
print(f"Training...\n", flush=True)

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

# 训前保存 tokenizer (防止训练中断丢失)
tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

for ep in range(start_ep, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    train_batches = num_batches_per_epoch // SUBSAMPLE
    pbar = tqdm(range(train_batches), desc=f"ep {ep}/{EPOCHS}")
    train_steps = resume_steps if ep == start_ep else 0
    skip_until = resume_steps if ep == start_ep else 0
    for bi in pbar:
        train_steps += 1
        if train_steps <= skip_until:
            continue
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)

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
        if train_steps % 500 == 0:
            avg_loss = total_loss / train_steps
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            att = model.get_att_rate()
            pbar.set_postfix(loss=f"{ce_loss.item():.3f}", ppl=f"{ppl:.1f}",
                           att=f"{att*100:.0f}%",
                           lr=f"{opt.param_groups[0]['lr']:.1e}")
            # 中间检查点 (防止 epoch 内崩溃)
            if train_steps % 2000 == 0:
                torch.save({
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "lr_scheduler": lr_scheduler.state_dict(),
                    "ep": ep, "ppl": ppl, "train_steps": train_steps,
                }, os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt"))
                tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

    avg_loss = total_loss / train_steps
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    att_rate = model.get_att_rate()
    print(f"ep {ep:2d}: loss={avg_loss:.4f} ppl={ppl:.1f} att={att_rate*100:.0f}% "
          f"lr={opt.param_groups[0]['lr']:.1e}")
    lr_scheduler.step()

    # checkpoint
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ep": ep, "ppl": ppl,
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt"))
    tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))

# final
torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_final.pt"))
tok.save(os.path.join(CKPT_DIR, f"{CKPT_NAME}"))
print(f"\nDone. Final ppl={ppl:.1f}")

# ── 参数量对标 ──
print(f"\nParams: {n_params:,} ({n_params/1e6:.1f}M)")
print(f"V1 full-rank: 14.2M → ppl 34.5")
print(f"V1 low-rank r=128: 11.6M → ppl 40.6")
