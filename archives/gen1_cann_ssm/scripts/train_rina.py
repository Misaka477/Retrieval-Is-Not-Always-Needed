"""
Train RINA with dual attractor basins and prediction-error-driven memory.

Tests if protected basins can retain information across longer gaps
than a single-state CANN.

Usage:
  python scripts/train_rina.py --gap 16
"""
import os, sys, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rina_cell import RINASimpleModel


def make_recall_data(n_seqs=1000, gap=8, n_keys=10):
    PAD = 0
    FILLER = 2 * n_keys + 1
    seq_len = gap + 3
    inputs, targets = [], []
    for _ in range(n_seqs):
        key = torch.randint(1, n_keys + 1, (1,)).item()
        val = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()
        seq = [key, val] + [FILLER] * gap + [key]
        tgt = [PAD] * (seq_len - 1) + [val]
        inputs.append(seq)
        targets.append(tgt)
    return torch.tensor(inputs), torch.tensor(targets)


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_normal", type=int, default=1024)
    p.add_argument("--n_protected", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_keys = 10
    vocab = 2 * n_keys + 3

    inputs, targets = make_recall_data(n_seqs=2000, gap=args.gap, n_keys=n_keys)
    seq_len = args.gap + 3

    model = RINASimpleModel(
        vocab_size=vocab,
        d_model=args.d_model,
        n_normal=args.n_normal,
        n_protected=args.n_protected,
        beta=args.beta,
        n_iter=args.n_iter,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"gap={args.gap}, vocab={vocab}, params={n_params:,}, seq_len={seq_len}")
    print(f"d_model={args.d_model}, n_normal={args.n_normal}, n_protected={args.n_protected}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_train = 1600

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i+args.batch_size]
            x = inputs[idx].to(device)
            y = targets[idx].to(device)

            logits, h_states = model.forward_with_error(x)

            ws = torch.ones(seq_len, device=device)
            ws[-1] = 50.0
            loss = (F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction='none')
                    .reshape(-1, seq_len) * ws).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                x_test = inputs[n_train:2000].to(device)
                y_test = targets[n_train:2000].to(device)
                logits, _ = model.forward_with_error(x_test)
                pred = logits[:, -1].argmax(dim=-1)
                acc = (pred == y_test[:, -1]).float().mean().item()
            print(f"  epoch {epoch}: loss={total_loss/100:.4f}, recall={acc*100:.0f}%")

    model.eval()
    with torch.no_grad():
        x_test = inputs[n_train:2000].to(device)
        y_test = targets[n_train:2000].to(device)
        logits, _ = model.forward_with_error(x_test)
        pred = logits[:, -1].argmax(dim=-1)
        acc = (pred == y_test[:, -1]).float().mean().item()
    print(f"\nFinal: gap={args.gap}, recall={acc*100:.1f}%")

    return acc


if __name__ == "__main__":
    train()
