"""LLaMA 3.2 1B baseline on 3 distributions (native tokenizer)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn.functional as F

device = "cuda"; SEQ, BS = 128, 4
CKPT = "D:/Software_Development/Project/models/Llama-3.2-1B"

print(f"Loading {CKPT}...")
m = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).to(device).eval()
tok = AutoTokenizer.from_pretrained(CKPT)
tok.pad_token = tok.eos_token
n = sum(p.numel() for p in m.parameters()) / 1e6
print(f"  {n:.0f}M params, vocab={m.config.vocab_size}")

def get_ids(text_gen, n_paras, desc, max_len=640):
    ids = []
    for i, t in enumerate(text_gen):
        if i >= n_paras: break
        enc = tok.encode(t, max_length=max_len, truncation=True)
        if len(enc) >= SEQ:
            ids.append(torch.tensor(enc, dtype=torch.long))
    ids = torch.cat(ids) if ids else torch.zeros(0, dtype=torch.long)
    print(f"  {desc}: {len(ids):,} tokens")
    return ids

def run(ids, name):
    nb = max(1, (len(ids) - 1) // (BS * SEQ)); tl = 0.0
    with torch.no_grad():
        for bi in range(nb):
            x = ids[bi * BS * SEQ : bi * BS * SEQ + BS * SEQ].view(BS, SEQ).to(device)
            o = m(x)
            lo = o.logits if hasattr(o, "logits") else o
            tl += F.cross_entropy(lo[:, :-1].reshape(-1, m.config.vocab_size), x[:, 1:].reshape(-1)).item()
    p = torch.exp(torch.tensor(tl / nb)).item()
    print(f"  {name:<20} {p:.1f}")
    return p

print("\nLoading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
txt_wt = [t["text"] for t in ds if len(t["text"]) > 100][:500]
wt = get_ids(txt_wt, 500, "WikiText-103")

print("Loading FineWeb (seed=999)...")
bank = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).shuffle(seed=999)
bank_iter = iter(bank)
txt_fw = [next(bank_iter)["text"] for _ in range(1000)]
fw = get_ids(txt_fw, 500, "FineWeb (unseen)")

print("Loading StarCoder...")
code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
code_iter = iter(code)
txt_cd = [next(code_iter)["content"] for _ in range(1000)]
cd = get_ids(txt_cd, 500, "Code (zero-shot)")

print(f"\n{'Model':<35} {'WikiText':>10} {'FineWeb':>10} {'Code':>10}")
print("-" * 67)
print(f"{'LLaMA 3.2 1B':<35} {run(wt,'WT'):>10.1f} {run(fw,'FW'):>10.1f} {run(cd,'CD'):>10.1f}")
