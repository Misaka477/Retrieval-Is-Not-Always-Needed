"""RINA vs GPT-2 124M: ppl comparison across 3 distributions + code zero-shot."""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from transformers import GPT2LMHeadModel
from tqdm import tqdm

device = "cuda"; V = 4096; SEQ, BS = 64, 8

def eval_model(m, ids, name, V=V):
    vocab_size = getattr(m, 'config', None).vocab_size if hasattr(m, 'config') else V
    if hasattr(m, 'config'): V = vocab_size
    num_batches = max(1, (len(ids) - 1) // (BS * SEQ))
    total_loss = 0.0
    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc=name[:16], leave=False):
            start = bi * BS * SEQ
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            out = m(x)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    return torch.exp(torch.tensor(total_loss / num_batches)).item()

# Load models
print("Loading RINA checkpoints...")
rina_checkpoints = {
    "RINA (FineWeb ep3)": "checkpoints/fineweb_resume.pt",
    "RINA (code-seq256)": "checkpoints/code_seq256_resume.pt",
}
rina_models = {}
for name, path in rina_checkpoints.items():
    sd = torch.load(path, map_location=device, weights_only=False)
    m = TemporalSNNModel(V, d_model=840, n_patterns=4096, beta=0.5).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.eval()
    rina_models[name] = m

print("Loading GPT-2 124M...")
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
print(f"  GPT-2 vocab: {gpt2.config.vocab_size}")

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

def tokenize(texts, desc):
    ids = [torch.tensor(tok.encode(t)[:min(len(tok.encode(t)), 640)], dtype=torch.long) for t in texts if len(tok.encode(t)) >= 64]
    ids = torch.cat(ids) if ids else torch.zeros(0, dtype=torch.long)
    print(f"  {desc}: {len(ids):,} tokens")
    return ids

# WikiText-103
print("\nWikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
wt = tokenize([t for t in ds["text"] if len(t) > 100][:500], "WikiText")

# FineWeb unseen
print("FineWeb unseen (seed=999, 500 paragraphs)...")
bank = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).shuffle(seed=999)
fw = tokenize([next(iter(bank))["text"] for _ in range(500)], "FineWeb")

# Code zero-shot (StarCoder)
print("Code (StarCoder, zero-shot)...")
ds_code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
code = tokenize([next(iter(ds_code))["content"] for _ in range(500)], "Code")

# Run
data = {"WikiText-103": wt, "FineWeb (unseen)": fw, "Code (zero-shot)": code}
print(f"\n{'Model':<28} {'WikiText':>10} {'FineWeb':>10} {'Code':>10} {'Params':>10}")
print("-" * 70)

results = []
for name, m in {**rina_models, "GPT-2 124M": gpt2}.items():
    Vm = V if "RINA" in name else gpt2.config.vocab_size
    row = [name]
    for dname, dids in data.items():
        p = eval_model(m, dids, name[:10], Vm)
        row.append(f"{p:.1f}")
    pm = sum(p.numel() for p in m.parameters()) / 1e6
    row.append(f"{pm:.0f}M")
    print(f"{name:<28} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
    results.append(row)
