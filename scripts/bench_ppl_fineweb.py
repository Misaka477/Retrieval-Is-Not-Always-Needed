"""FineWeb validation ppl — 用未见过数据检测过拟合。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; V, SEQ, BS = 4096, 64, 8
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

ckpt_sources = [
    ("WikiText baseline", "checkpoints/cann_snn15m_v2_slot_ep12.pt"),
    ("FineWeb ep1", "checkpoints/fineweb_ep1.pt"),
    ("FineWeb ep2", "checkpoints/fineweb_ep2.pt"),
    ("FineWeb ep3 (seed=43)", "checkpoints/fineweb_resume.pt"),
]

# Build model once, reload weights per checkpoint
model = TemporalSNNModel(V, d_model=840, n_patterns=4096, beta=0.5,
                          attract_every=2, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.8).to(device)
model.eval()

# Load unseen FineWeb data (streaming, different seed -> different subset)
print("Loading unseen FineWeb data (500 paragraphs)...", flush=True)
bank = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
bank = bank.shuffle(buffer_size=10000, seed=999)  # different seed from training (42)
texts = []
for i, sample in enumerate(bank):
    if i >= 500: break
    texts.append(sample["text"])

# Tokenize
all_ids = []
for t in tqdm(texts, desc="tokenizing"):
    ids = tok.encode(t).ids
    if len(ids) >= SEQ:
        all_ids.append(torch.tensor(ids[:min(len(ids), SEQ * 100)], dtype=torch.long))
ids = torch.cat(all_ids) if all_ids else torch.zeros(0, dtype=torch.long)
num_batches = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  tokens: {len(ids):,}, batches: {num_batches}")

print(f"\n{'Model':<20} {'ppl':>8}")
print("-" * 30)
for name, ckpt_path in ckpt_sources:
    if not os.path.exists(ckpt_path):
        print(f"{name:<20} {'no checkpoint':>8}")
        continue
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    total_loss = 0.0
    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc=name, leave=False):
            start = bi * BS * SEQ
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"{name:<20} {ppl:>8.2f}")
