"""Compare code-seq256 vs slot checkpoint on WikiText-103."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; V, SEQ, BS = 4096, 64, 8
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t for t in ds["text"] if len(t) > 100][:2000]
all_ids = [torch.tensor(tok.encode(t).ids[:min(len(tok.encode(t).ids), 640)], dtype=torch.long) for t in tqdm(texts, desc="tokenizing") if len(tok.encode(t).ids) >= 64]
ids = torch.cat(all_ids) if all_ids else torch.zeros(0, dtype=torch.long)
num_batches = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  tokens: {len(ids):,}, batches: {num_batches}")

checkpoints = [
    ("Slot checkpoint (best general)", "checkpoints/cann_snn15m_v2_slot_ep13.pt"),
    ("Code-seq256 (last code)", "checkpoints/code_seq256_resume.pt"),
]
results = []
for name, path in checkpoints:
    if not os.path.exists(path):
        results.append((name, -1))
        continue
    sd = torch.load(path, map_location=device, weights_only=False)
    m = TemporalSNNModel(V, d_model=840, n_patterns=4096, beta=0.5,
                          attract_every=2, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.eval()
    total_loss = 0.0
    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc=name[:20], leave=False):
            start = bi * BS * SEQ
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
    results.append((name, ppl))
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'Model':<35} {'WikiText ppl':>15}")
print("-" * 52)
for name, ppl in results:
    print(f"{name:<35} {ppl:>15.2f}")
