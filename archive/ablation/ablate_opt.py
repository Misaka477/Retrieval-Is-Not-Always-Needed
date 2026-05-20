"""Opt + Scheduler 消融: 验证 weight_decay/LR scheduler 对 ppl 的影响。dm=256, 3 epoch."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

import torch, torch.nn.functional as F, time
torch.manual_seed(42)
from modules.temporal_snn_cell import TemporalSNNModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

print("Loading...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw = [t for t in ds["text"] if len(t) > 100][:20000]

print("Training BPE...", flush=True)
tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trn = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>"])
tok.train_from_iterator(raw[:8000], trn); tok.add_special_tokens(["<pad>"])
V = tok.get_vocab_size()
print(f"  vocab: {V}", flush=True)

print("Tokenizing...", flush=True)
il = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw if len(tok.encode(t).ids) > 64]
ids = torch.cat(il)[:1000000]
print(f"  tokens: {len(ids):,}", flush=True)

DM, NP, SEQ, BS, AE = 256, 1024, 64, 8, 2
EPOCHS, LR, SS = 3, 3e-4, 2
TH, HL, INH = 1.0, 0.01, 0.8
nt = (len(ids) - 1) // (BS * SEQ) // SS
print(f"  batches/epoch: {nt}", flush=True)

def train_one(name, wd, betas, lr_per_step):
    m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=AE,
                          error_threshold=TH, hebbian_lr=HL,
                          inhibition_threshold=INH).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=betas, weight_decay=wd)
    t_max = EPOCHS if not lr_per_step else EPOCHS * nt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        m.train(); tl = 0.0
        perm = torch.randperm(len(ids) - BS * SEQ)
        pbar = tqdm(range(nt), desc=f"{name} ep{ep}/{EPOCHS}", leave=False)
        for bi in pbar:
            s = perm[(bi * SS) % len(perm)]
            x = ids[s:s + BS * SEQ].view(BS, SEQ).to(device)
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); tl += loss.item()
            if lr_per_step: sched.step()
            if bi % 100 == 99:
                ppl = torch.exp(torch.tensor(tl / (bi + 1))).item()
                pbar.set_postfix(ppl=f"{ppl:.1f}")
        if not lr_per_step: sched.step()
        ppl = torch.exp(torch.tensor(tl / nt)).item()
        print(f"  {name} ep{ep}: ppl={ppl:.1f} att={m.get_att_rate()*100:.0f}% lr={opt.param_groups[0]['lr']:.1e}", flush=True)
    return ppl, time.time() - t0

print(f"\n{'='*60}")
print(f"Opt+Scheduler Ablation: dm={DM} np={NP} th={TH} ep={EPOCHS}")
print(f"{'='*60}")

configs = [
    ("SNN旧 (wd=0.1,b2=.95,step)", 0.1, (0.9, 0.95), True),
    ("V1配 (wd=.01,b2=.999,ep)", 0.01, (0.9, 0.999), False),
]

results = {}
for name, wd, betas, step in configs:
    print(f"\n-- {name} --", flush=True)
    ppl, elapsed = train_one(name, wd, betas, step)
    results[name] = (ppl, elapsed)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for name in [c[0] for c in configs]:
    p, t = results[name]
    print(f"  {name:35s} ppl={p:6.1f}  {t/60:4.1f}m")
