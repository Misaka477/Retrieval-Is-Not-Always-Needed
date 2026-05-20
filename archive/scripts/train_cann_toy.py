"""
Train minimal RINA toy model (CANN + state evolution) on simple tasks.

Task 1: Associative Recall (MQAR-style)
  Sequence: [key tokens] [filler] [key tokens] [query]
  Model must recall the value associated with the key.

Task 2: Next-token prediction on real text (Pride and Prejudice)
  Standard language modeling.

Usage:
  python scripts/train_cann_toy.py --task recall --epochs 50
  python scripts/train_cann_toy.py --task lm --epochs 10
"""
import os, sys, argparse, json, math
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_memory import CANNSimpleModel


def make_recall_data(n_seqs=1000, n_keys=20, seq_len=64, gap_min=8, gap_max=32):
    """
    Generate Associative Recall sequences.

    Each sequence:
      key1 val1 key2 val2 ... (at beginning)
      ... filler tokens ...
      key1 -> model should output val1

    Token ID layout:
      1 = <pad>
      2..n_keys+1 = keys
      n_keys+2..2*n_keys+1 = values
      2*n_keys+2 = filler

    Returns: (input_ids, target_ids)
    """
    PAD = 0
    FILLER_TOK = n_keys * 2 + 1

    all_inputs, all_targets = [], []
    for _ in range(n_seqs):
        pairs = {}
        seq = []
        n_pairs = torch.randint(2, 5, (1,)).item()
        used = set()
        for _ in range(n_pairs):
            k = torch.randint(0, n_keys, (1,)).item()
            while k in used:
                k = torch.randint(0, n_keys, (1,)).item()
            used.add(k)
            v = n_keys + torch.randint(0, n_keys, (1,)).item()
            pairs[k + 1] = v + 1
            seq.extend([k + 1, v + 1])

        gap = torch.randint(gap_min, gap_max, (1,)).item()
        filler = [FILLER_TOK] * gap
        seq.extend(filler)

        query_key = list(pairs.keys())[0]
        expected_val = pairs[query_key]
        seq.append(query_key)

        input_ids = seq
        target_ids = [PAD] * (len(seq) - 1) + [expected_val]

        pad_len = seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [PAD] * pad_len
            target_ids = target_ids + [PAD] * pad_len
        else:
            input_ids = input_ids[:seq_len]
            target_ids = target_ids[:seq_len]

        all_inputs.append(input_ids)
        all_targets.append(target_ids)

    return torch.tensor(all_inputs), torch.tensor(all_targets)


def get_pride_data(tok_path=None, seq_len=128):
    """Load Pride and Prejudice text and tokenize with simple char-level tokens."""
    import collections
    text_path = tok_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.pardir, "RINA_Core", "scripts", "evaluation", "_pride.txt"
    )

    if os.path.exists(text_path):
        with open(text_path, "r", encoding="latin-1") as f:
            text = f.read()
    else:
        text = (
            "It is a truth universally acknowledged, that a single man in possession "
            "of a good fortune, must be in want of a wife. However little known the "
            "feelings or views of such a man may be on his first entering a "
            "neighbourhood, this truth is so well fixed in the minds of the surrounding "
            "families, that he is considered the rightful property of some one or other "
            "of their daughters."
        ) * 50

    chars = sorted(list(set(text)))
    stoi = {ch: i + 3 for i, ch in enumerate(chars)}
    stoi['<pad>'] = 0
    stoi['<bos>'] = 1
    stoi['<eos>'] = 2
    itos = {i: ch for ch, i in stoi.items()}

    data = [stoi.get(ch, stoi['<pad>']) for ch in text]
    data = torch.tensor(data)

    n = len(data)
    inputs = []
    targets = []
    stride = seq_len // 2
    for i in range(0, n - seq_len - 1, stride):
        x = data[i:i+seq_len]
        y = data[i+1:i+seq_len+1]
        inputs.append(x)
        targets.append(y)

    return torch.stack(inputs), torch.stack(targets), len(stoi)


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["recall", "lm"], default="recall")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_patterns", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save", default="checkpoints/cann_toy.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.task == "recall":
        inputs, targets = make_recall_data(n_seqs=2000, seq_len=args.seq_len)
        vocab_size = targets.max().item() + 2
        n_test = 200
        train_in, test_in = inputs[:-n_test], inputs[-n_test:]
        train_tg, test_tg = targets[:-n_test], targets[-n_test:]
        print(f"Recall data: {len(train_in)} train, {len(test_in)} test, vocab={vocab_size}")
    else:
        inputs, targets, vocab_size = get_pride_data(seq_len=args.seq_len)
        n_test = int(len(inputs) * 0.1)
        train_in, test_in = inputs[:-n_test], inputs[-n_test:]
        train_tg, test_tg = targets[:-n_test], targets[-n_test:]
        print(f"LM data: {len(train_in)} train, {len(test_in)} test, vocab={vocab_size}")

    model = CANNSimpleModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_patterns=min(args.n_patterns, vocab_size * 4),
        beta=args.beta,
        n_iter=args.n_iter,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"  d_model={args.d_model}, n_patterns={args.n_patterns}, beta={args.beta}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        perm = torch.randperm(len(train_in))
        for i in range(0, len(train_in), args.batch_size):
            idx = perm[i:i+args.batch_size]
            x = train_in[idx].to(device)
            y = train_tg[idx].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        perplexity = math.exp(avg_loss)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            test_loss = 0
            n_test_batches = 0
            with torch.no_grad():
                for i in range(0, len(test_in), args.batch_size):
                    x = test_in[i:i+args.batch_size].to(device)
                    y = test_tg[i:i+args.batch_size].to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                    test_loss += loss.item()
                    n_test_batches += 1
            test_ppl = math.exp(test_loss / n_test_batches)
            print(f"Epoch {epoch:3d} | train loss {avg_loss:.4f} (ppl {perplexity:.2f}) | test ppl {test_ppl:.2f}")

    os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else ".", exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    train()
