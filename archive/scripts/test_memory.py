"""
Test CANN memory recall: NIAH-style evaluation.

Given a sequence with an embedded "needle" (e.g. "The password is KILO42"),
can the CANN state remember it after processing thousands of tokens?

Tests:
  1. State-based recall: does the final state contain needle info?
  2. Generation: after the sequence, can the model output the needle?
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_memory import CANNSimpleModel


def build_haystack(needle_pos, context_len, needle_ids, haystack_ids):
    """Insert needle into repeated haystack at position needle_pos."""
    seq = []
    rep_idx = 0
    while len(seq) < context_len:
        if rep_idx == needle_pos:
            seq.extend(needle_ids)
        else:
            seq.extend(haystack_ids)
        rep_idx += 1
    return seq[:context_len]


def test_state_recall(model, vocab_size, device):
    """
    Test 1: Encode sequence with needle, then probe the final state.
    The final state should contain information about the needle.
    We probe by projecting the state through the model head and checking
    if the predicted logits include needle-related tokens.
    """
    needle_str = "The secret password is KILO42. "
    query_str = "The password is "
    context_lens = [128, 256, 512]
    depths = [0.25, 0.5, 0.75]

    stoi = {ch: i for i, ch in enumerate(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-'\"\n;:()"
    )}
    stoi['<pad>'] = 0
    stoi['<bos>'] = 1
    stoi['<eos>'] = 2
    stoi['.'] = stoi.get('.', 3)

    def encode(text):
        return [stoi.get(ch, stoi.get('<pad>', 0)) for ch in text]

    needle_ids = encode(needle_str)
    haystack_ids = encode("The grass is green. The sky is blue. ")
    query_ids = encode(query_str)

    results = []
    model.eval()

    for ctx_len in context_lens:
        for depth in depths:
            needle_pos = max(0, int(ctx_len * depth / len(haystack_ids)))
            seq = build_haystack(needle_pos, ctx_len, needle_ids, haystack_ids)
            seq = seq + query_ids

            if max(seq) >= vocab_size:
                continue

            x = torch.tensor([seq], device=device)

            with torch.no_grad():
                logits, final_state = model(x, return_state=True)

            # Probe final state: what does the model predict next?
            state_logits = model.head(model.state_norm(final_state))
            probs = F.softmax(state_logits, dim=-1)
            top5 = probs.topk(5)

            predicted_chars = []
            for idx in top5.indices[0]:
                for ch, i in stoi.items():
                    if i == idx.item():
                        predicted_chars.append(ch)
                        break

            has_needle = any("KILO42" in "".join(predicted_chars) for _ in [1])
            needle_token_found = top5.indices[0].tolist()

            print(f"ctx={ctx_len:4d} depth={depth:.2f} | "
                  f"top-5: {predicted_chars} | "
                  f"{'✅' if predicted_chars else '❌'}")

            results.append({
                "context_len": ctx_len,
                "depth": depth,
                "needle_pos_tok": needle_pos * len(haystack_ids),
                "top5_tokens": predicted_chars,
                "pass": any("KILO" in c for c in predicted_chars)
            })

    return results


def test_generation(model, vocab_size, device):
    """
    Test 2: Generate text after seeing the needle.
    More realistic: run the model autoregressively.
    """
    needle_str = "The secret password is KILO42. "
    query_str = " I just told you the password. The password is"
    haystack_str = "The grass is green. The sky is blue. "

    stoi = {ch: i for i, ch in enumerate(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-'\"\n;:()"
    )}
    stoi['<pad>'] = 0
    stoi['<bos>'] = 1
    stoi['<eos>'] = 2

    def encode(text):
        return [stoi.get(ch, stoi.get('<pad>', 0)) for ch in text]

    def decode(ids):
        itos = {i: ch for ch, i in stoi.items()}
        return "".join(itos.get(i, '') for i in ids)

    needle_ids = encode(needle_str)
    haystack_ids = encode(haystack_str)
    query_ids = encode(query_str)

    context_lens = [128, 256]
    depths = [0.25, 0.5, 0.75]
    max_new = 20

    results = []
    model.eval()

    for ctx_len in context_lens:
        for depth in depths:
            needle_pos = max(0, int(ctx_len * depth / len(haystack_ids)))
            seq = build_haystack(needle_pos, ctx_len, needle_ids, haystack_ids)
            prompt_ids = seq + query_ids

            if max(prompt_ids) >= vocab_size:
                continue

            prompt_t = torch.tensor([prompt_ids], device=device)
            generated = prompt_ids[:]

            with torch.no_grad():
                for _ in range(max_new):
                    x = torch.tensor([generated], device=device)
                    logits = model(x)
                    next_logit = logits[0, -1]
                    next_id = next_logit.argmax().item()
                    generated.append(next_id)

            gen_text = decode(generated[len(prompt_ids):])
            has_needle = "KILO42" in gen_text

            print(f"ctx={ctx_len:4d} depth={depth:.2f} | "
                  f"gen: {gen_text[:40]:40s} | "
                  f"{'✅' if has_needle else '❌'}")

            results.append({
                "context_len": ctx_len,
                "depth": depth,
                "generated": gen_text,
                "pass": has_needle
            })

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/cann_toy.pt")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_patterns", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--test", choices=["state", "generation", "both"], default="both")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = 70
    stoi = {ch: i for i, ch in enumerate(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-'\"\n;:()"
    )}
    vocab_size = max(vocab_size, max(stoi.values()) + 1) + 3

    model = CANNSimpleModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_patterns=args.n_patterns,
        beta=args.beta,
        n_iter=args.n_iter,
    ).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded: {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Using untrained model (random).")

    all_results = {}
    if args.test in ("state", "both"):
        print("\n=== State Recall Test ===")
        sr = test_state_recall(model, vocab_size, device)
        all_results["state_recall"] = sr
        state_pass = sum(1 for r in sr if r["pass"])
        print(f"State recall: {state_pass}/{len(sr)}")

    if args.test in ("generation", "both"):
        print("\n=== Generation Test ===")
        gr = test_generation(model, vocab_size, device)
        all_results["generation"] = gr
        gen_pass = sum(1 for r in gr if r["pass"])
        print(f"Generation: {gen_pass}/{len(gr)}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved: {args.json}")


if __name__ == "__main__":
    main()
