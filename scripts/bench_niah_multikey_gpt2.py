"""GPT-2 124M on multi-key NIAH (seq=1024, native training length)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda"; SEQ = 1024; BS = 1
N_KEYS = 3; GAP = 128
V = 4096; KEYS = list(range(1, N_KEYS + 1)); VALS = list(range(101, 101 + N_KEYS))

print(f"GPT-2 124M — multi-key NIAH ({N_KEYS} keys, gap={GAP}, seq={SEQ})")

m = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tok.pad_token = tok.eos_token
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"  {n:.0f}M params")

tok4096 = __import__("tokenizers", fromlist=["Tokenizer"]).Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"]) > 200][:2000]

def make_batch(bs, seq, n_keys, gap):
    x = torch.randint(V, (bs, seq))
    for b in range(bs):
        txt = random.choice(texts)
        ids = tok4096.encode(txt).ids
        # Fill with real text
        for t in range(min(seq, len(ids))):
            x[b, t] = ids[t] if t < len(ids) else 0
        for k in range(n_keys):
            kv_pos = (seq // (n_keys + 1)) * (k + 1)
            if kv_pos + gap < seq:
                x[b, kv_pos] = KEYS[k]
                x[b, kv_pos + 1] = VALS[k]
                x[b, -1 - k] = KEYS[k]
    return x

print("Zero-shot evaluation (no fine-tune)...")
m.eval(); correct, total = 0, 0
with torch.no_grad():
    for _ in range(200):
        x = make_batch(1, SEQ, N_KEYS, GAP).to(device)
        out = m(x)
        logits = out.logits if hasattr(out, "logits") else out
        for k in range(N_KEYS):
            total += 1
            if logits[0, -1 - k].argmax().item() == VALS[k]:
                correct += 1
acc = correct / total

print(f"\n  GPT-2 124M (seq=1024) multi-key ({N_KEYS}, gap={GAP}): {acc*100:.0f}%")
print(f"  RINA 15M (seq=64):                                      100%")
print(f"  GPT-2 15M (seq=64):                                      36%")
