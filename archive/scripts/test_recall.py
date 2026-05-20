"""
Test trained CANN model on Associative Recall task.

Generates fresh test sequences and checks if the model
can correctly recall the value associated with each key.
"""
import os, sys, argparse, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_memory import CANNSimpleModel


def make_recall_data(n_seqs=1000, n_keys=20, seq_len=64, gap_min=8, gap_max=32):
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


@torch.no_grad()
def test_recall():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/cann_toy.pt")
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_patterns", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_test", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs, targets = make_recall_data(n_seqs=args.n_test, seq_len=args.seq_len)
    vocab_size = targets.max().item() + 2
    _, test_in = inputs[:1], inputs
    _, test_tg = targets[:1], targets

    model = CANNSimpleModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_patterns=min(args.n_patterns, vocab_size * 4),
        beta=args.beta,
        n_iter=args.n_iter,
    ).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded: {args.checkpoint}")
    else:
        print("No checkpoint, using untrained model")

    model.eval()
    correct = 0
    total = 0

    for i in range(len(test_in)):
        x = test_in[i:i+1].to(device)
        y = test_tg[i:i+1].to(device)
        logits = model(x)

        pred = logits[0, -1].argmax().item()
        actual = y[0, -1].item()

        if pred == actual:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"\nAssociative Recall Accuracy: {correct}/{total} ({acc:.1f}%)")

    if acc < 50:
        print("\nSample failures:")
        failures = 0
        for i in range(len(test_in)):
            x = test_in[i:i+1].to(device)
            y = test_tg[i:i+1].to(device)
            logits = model(x)
            pred = logits[0, -1].argmax().item()
            actual = y[0, -1].item()
            if pred != actual and failures < 5:
                print(f"  seq {i}: pred={pred}, actual={actual}")
                failures += 1

    return acc


if __name__ == "__main__":
    test_recall()
