"""
Quick seq=1024 evaluation: code-seq256 on code, FineWeb ep3 on FineWeb.
Limited to 20 batches per model for speed.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel

device = "cuda"; SEQ, BS = 1024, 1
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

def eval_on_data(ckpt, texts, name, max_batches=20):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.eval()
    ids = []
    for t in texts:
        enc = tok.encode(t).ids
        if len(enc) >= SEQ:
            ids.append(torch.tensor(enc[:SEQ], dtype=torch.long))
            if len(ids) >= max_batches: break
    if not ids: return -1
    ids = torch.cat(ids)
    nb = (len(ids) - 1) // (BS * SEQ)
    tl = 0.0
    with torch.no_grad():
        for bi in range(nb):
            x = ids[bi * BS * SEQ : bi * BS * SEQ + BS * SEQ].view(BS, SEQ).to(device)
            tl += F.cross_entropy(m(x)[:, :-1].reshape(-1, 4096), x[:, 1:].reshape(-1)).item()
    return torch.exp(torch.tensor(tl / nb)).item()

print("Loading data (500 samples each, streaming)...")
code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
cd = [next(iter(code))["content"] for _ in range(500)]
bank = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).shuffle(seed=999)
fw = [next(iter(bank))["text"] for _ in range(500)]

print(f"\n{'Checkpoint':<30} {'Data':<15} {'ppl@seq=1024':>15}")
print("-" * 62)
p1 = eval_on_data("checkpoints/code_seq256_resume.pt", cd, "code-seq256")
print(f"{'code-seq256':<30} {'StarCoder':<15} {p1:>15.1f}")
p2 = eval_on_data("checkpoints/fineweb_resume.pt", fw, "FineWeb ep3")
print(f"{'FineWeb ep3':<30} {'FineWeb':<15} {p2:>15.1f}")
print()
print(f"Reference code-seq256 @ seq=64: ~5.03")
print(f"Reference FineWeb ep3 @ seq=64:  ~45.73")
