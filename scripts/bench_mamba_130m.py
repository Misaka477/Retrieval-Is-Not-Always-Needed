"""Mamba-130M on WikiText-103 — pure SSM baseline."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import MambaForCausalLM, AutoTokenizer
import torch, torch.nn.functional as F

device = "cuda"; SEQ, BS = 1024, 1

print("Loading Mamba-130M...")
m = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.float16).to(device).eval()
tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tok.pad_token = tok.eos_token
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"  {n:.0f}M params")

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
print(f"\nMamba-130M (native tokenizer, seq=1024): WikiText ppl={ppl:.1f}")
print()
print("Comparison:")
print(f"  RINA 15M (4K vocab, seq=64):             34.6")
print(f"  GPT-2 124M (50K vocab, seq=1024):        25.4")
print(f"  LLaMA 3.2 1B (128K vocab, seq=1024):    11.4")
print(f"  Mamba-130M (native vocab, seq=1024):    {ppl:.1f}")
