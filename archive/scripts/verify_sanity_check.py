"""Sanity check: TinyLLaMA 1.1B on WikiText-103 — verify evaluation pipeline."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn.functional as F

device = "cuda"; SEQ, BS = 1024, 1

print("Loading TinyLLaMA 1.1B...")
m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).to(device).eval()
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.pad_token = tok.eos_token
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"  {n:.0f}M params, vocab={m.config.vocab_size}")

ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:100]
ids_list = []
for t in texts:
    enc = tok.encode(t, max_length=SEQ, truncation=True)
    if len(enc) >= 64:
        ids_list.append(torch.tensor(enc, dtype=torch.long))
ids = torch.cat(ids_list) if ids_list else torch.zeros(0, dtype=torch.long)
nb = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  tokens: {len(ids):,}, batches: {nb}")

tl = 0.0
with torch.no_grad():
    for bi in range(nb):
        x = ids[bi * BS * SEQ : bi * BS * SEQ + BS * SEQ].view(BS, SEQ).to(device)
        o = m(x); lo = o.logits if hasattr(o, "logits") else o
        tl += F.cross_entropy(lo[:, :-1].reshape(-1, m.config.vocab_size), x[:, 1:].reshape(-1)).item()
        if bi % 2 == 0:
            print(f"  batch {bi+1}/{nb}: running loss={tl/(bi+1):.3f}", flush=True)

ppl = torch.exp(torch.tensor(tl / nb)).item()
print(f"\nTinyLLaMA 1.1B (native tokenizer, seq=1024): WikiText ppl={ppl:.1f}")
print(f"(Published TinyLLaMA WikiText ppl is ~12-14. If this is in range, pipeline is correct.)")
