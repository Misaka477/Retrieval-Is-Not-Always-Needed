"""
Test CANN long-range memory retention.

NIAH-style: embed a "secret number" at the beginning of a long sequence,
then test if the model can recall it from the final state after processing
many filler tokens.

Also tests: does accuracy degrade as sequence length increases?
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_memory import CANNSimpleModel


def make_niah_data(n_seqs=200, seq_len=128, needle_pos=0, n_keys=10):
    """
    NIAH-style sequences.

    Structure:
      [key=NEEDLE] [value=RAND] [filler...] [key=NEEDLE] [query=key]
      Model must recall the value from the beginning.

    Token layout:
      0 = PAD
      1..n_keys = keys
      n_keys+1..2*n_keys = values
      2*n_keys+1 = filler
    """
    PAD = 0
    FILLER_TOK = 2 * n_keys + 1
    all_inputs, all_targets, info = [], [], []

    for _ in range(n_seqs):
        needle_key = torch.randint(1, n_keys + 1, (1,)).item()
        needle_val = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()

        seq = [needle_key, needle_val]

        gap = seq_len - 3
        filler = [FILLER_TOK] * gap
        seq.extend(filler)

        seq.append(needle_key)

        target = [PAD] * (seq_len - 1) + [needle_val]
        input_ids = seq
        target_ids = target

        all_inputs.append(input_ids)
        all_targets.append(target_ids)
        info.append({"needle_key": needle_key, "needle_val": needle_val})

    return torch.tensor(all_inputs), torch.tensor(all_targets), info


def test():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_patterns", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--train_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_test", type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test across increasing sequence lengths
    seq_lens = [16, 32, 64, 128, 256, 512]
    results = []

    for seq_len in seq_lens:
        print(f"=== seq_len={seq_len} ===")

        inputs, targets, info = make_niah_data(
            n_seqs=500 + seq_len, seq_len=seq_len, needle_pos=0, n_keys=10
        )
        vocab_size = targets.max().item() + 2
        n_train = len(inputs) - args.n_test

        train_in, train_tg = inputs[:n_train], targets[:n_train]
        test_in, test_tg = inputs[n_train:], targets[n_train:]
        test_info = info[n_train:]

        model = CANNSimpleModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_patterns=min(args.n_patterns, vocab_size * 4),
            beta=args.beta,
            n_iter=args.n_iter,
        ).to(device)

        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
            print(f"Loaded checkpoint")
        else:
            # Train from scratch
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
            for epoch in range(args.train_epochs):
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
                if epoch % 10 == 9:
                    print(f"  epoch {epoch}: loss {total_loss/n_batches:.4f}")
            print(f"  trained")

        # Test: last-token recall accuracy
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(len(test_in)):
                x = test_in[i:i+1].to(device)
                y = test_tg[i:i+1].to(device)
                logits = model(x)
                pred = logits[0, -1].argmax().item()
                actual = y[0, -1].item()
                if pred == actual:
                    correct += 1

        acc = correct / len(test_in) * 100
        print(f"  Recall accuracy: {correct}/{len(test_in)} ({acc:.1f}%)\n")
        results.append({"seq_len": seq_len, "acc": acc, "correct": correct, "total": len(test_in)})

    print("\n=== Summary ===")
    for r in results:
        bar = "#" * int(r["acc"] / 5)
        print(f"  seq_len={r['seq_len']:4d}: {r['correct']:3d}/{r['total']:3d} ({r['acc']:5.1f}%) {bar}")


if __name__ == "__main__":
    test()
