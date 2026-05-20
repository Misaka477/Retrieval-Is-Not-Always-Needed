"""Threshold sweep: 验证 att 松动能改善 ppl。
5 组 threshold, dm=256, np=1024, 3 epoch, ~1M tokens."""
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
HL, INH = 0.01, 0.8
nt = (len(ids) - 1) // (BS * SEQ) // SS
print(f"  batches/epoch: {nt}", flush=True)

def train_one(error_threshold):
    m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=AE,
                          error_threshold=error_threshold,
                          hebbian_lr=HL, inhibition_threshold=INH).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        m.train(); tl = 0.0
        perm = torch.randperm(len(ids) - BS * SEQ)
        pbar = tqdm(range(nt), desc=f"th={error_threshold} ep{ep}/{EPOCHS}", leave=False)
        for bi in pbar:
            s = perm[(bi * SS) % len(perm)]
            x = ids[s:s + BS * SEQ].view(BS, SEQ).to(device)
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); tl += loss.item()
            if bi % 100 == 99:
                ppl = torch.exp(torch.tensor(tl / (bi + 1))).item()
                pbar.set_postfix(ppl=f"{ppl:.1f}")
        ppl = torch.exp(torch.tensor(tl / nt)).item()
        att = m.get_att_rate()
        print(f"  th={error_threshold} ep{ep}: ppl={ppl:.1f} att={att*100:.0f}%", flush=True)
    return ppl, att, time.time() - t0

print(f"\n{'='*60}")
print(f"Threshold Sweep: dm={DM} np={NP} ep={EPOCHS}")
print(f"{'='*60}")

thresholds = [0.3, 0.5, 0.7, 1.0, -1.0]
results = {}
for th in thresholds:
    name = f"th={th:.1f}" if th >= 0 else "always"
    print(f"\n-- {name} --", flush=True)
    ppl, att, elapsed = train_one(th)
    results[name] = (ppl, att, elapsed)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'Threshold':>10} {'PPL':>8} {'Att':>7} {'Time':>8}")
print("-" * 38)
for th in thresholds:
    name = f"th={th:.1f}" if th >= 0 else "always"
    p, a, t = results[name]
    print(f"{name:>10} {p:8.1f} {a*100:6.0f}% {t/60:7.1f}m")
