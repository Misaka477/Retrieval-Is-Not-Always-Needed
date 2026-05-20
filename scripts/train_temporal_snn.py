"""
Temporal SNN training — 对比不同 error_threshold 下的 ppl 和 attractor 调用率。

训小模型 (dm=128, np=512) 在 WikiText-2 小切片上，3 epoch 快速对比。
"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from modules.temporal_snn_cell import TemporalSNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 加载 tokenizer ──
tok_path = os.path.join(_ROOT, "checkpoints", "cann_15m_wt103-vocab.json")
tok = Tokenizer.from_file(tok_path)
V = tok.get_vocab_size()
print(f"Tokenizer: vocab={V}")

# ── 加载数据 ──
data_path = os.path.join(_ROOT, "data", "wikitext-2")
if not os.path.exists(data_path):
    data_path = os.path.join(_ROOT, "data", "wikitext-103")

import glob
txt_files = glob.glob(os.path.join(data_path, "*.txt")) + glob.glob(os.path.join(data_path, "*.raw"))
if not txt_files:
    # fallback: generate synthetic text
    import random
    random.seed(42)
    words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park", "a", "is", "was",
             "and", "or", "but", "with", "for", "from", "to", "of"]
    text = " ".join(random.choices(words, k=20000))
else:
    text = ""
    for f in txt_files[:1]:
        with open(f, "r", encoding="utf-8") as fh:
            text += fh.read()[:2000000]

ids = tok.encode(text).ids
ids = ids[:min(len(ids), 500000)]
print(f"Tokens: {len(ids)}")

# ── 配置 ──
DM = 128
NP = 512
SEQ = 128  # 更长序列才能体现 attractor 差异
BS = 8
EPOCHS = 3

# 生成 batches
data_t = torch.tensor(ids)
num_batches = (len(data_t) - 1) // (BS * SEQ)
print(f"Batches/epoch: {num_batches}")


def train_model(error_threshold, hebbian_lr=0.0, inhibition_threshold=0.0):
    model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                              attract_every=2, error_threshold=error_threshold,
                              hebbian_lr=hebbian_lr,
                              inhibition_threshold=inhibition_threshold).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()

    for ep in range(EPOCHS):
        total_loss = 0.0
        for bi in range(num_batches):
            start = (bi * BS * SEQ) % (len(data_t) - BS * SEQ)
            x = data_t[start:start + BS * SEQ].view(BS, SEQ).to(device)

            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                x[:, 1:].reshape(-1),
            )
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if bi % 200 == 199:
                ppl = torch.exp(torch.tensor(total_loss / (bi + 1))).item()
                hb = model.get_hebb_rate() if hebbian_lr > 0 else 0
                print(f"  th={error_threshold:.1f} hb_lr={hebbian_lr:.4f} "
                      f"ep={ep+1}/{EPOCHS} batch={bi+1}/{num_batches} "
                      f"ppl={ppl:.1f} att={model.get_att_rate()*100:.0f}% "
                      f"hb={hb*100:.0f}%")

        ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
        hb = model.get_hebb_rate() if hebbian_lr > 0 else 0
        print(f"  th={error_threshold:.1f} hb_lr={hebbian_lr:.4f} "
              f"ep={ep+1}: ppl={ppl:.1f} "
              f"att={model.get_att_rate()*100:.0f}% hb={hb*100:.0f}%")

    elapsed = time.time() - t0
    return model, ppl, elapsed


# ── 训练对比 ──
print(f"\n{'='*60}")
print(f"Training: dm={DM}, np={NP}, seq={SEQ}, bs={BS}, ep={EPOCHS}")
print(f"{'='*60}")

results = {}
configs = [
    (0.5, 0.0, 0.0, "th=0.5"),
    (0.5, 0.01, 0.0, "th=0.5+Hebb"),
    (0.5, 0.01, 0.8, "th=0.5+Hebb+Inhib"),
    (-1.0, 0.0, 0.0, "always"),
]
for th, hb_lr, inhib_th, label in configs:
    print(f"\n── {label} ──")
    model, ppl, elapsed = train_model(th, hebbian_lr=hb_lr, inhibition_threshold=inhib_th)
    att_rate = model.get_att_rate()
    results[label] = {"ppl": ppl, "att_rate": att_rate, "time": elapsed}

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Config':>18} {'PPL':>8} {'Att%':>7} {'Time':>8}")
print("-" * 46)
for label in [c[3] for c in configs]:
    r = results[label]
    print(f"{label:>18} {r['ppl']:8.1f} {r['att_rate']*100:6.0f}% {r['time']/60:7.1f}m")
