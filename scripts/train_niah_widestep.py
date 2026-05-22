"""
Quick NIAH wide-gap fine-tune — load code-seq256 checkpoint, train with gap=32-128 
NIAH samples for 200 steps, test gap=64/128 recall improvement.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42); random.seed(42)
V, DM, NP, V = 4096, 840, 4096, 4096
SEQ = 256
CKPT = "checkpoints/code_seq256_resume.pt"

print(f"Loading {CKPT}...")
sd = torch.load(CKPT, map_location=device, weights_only=False)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=2, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.0,
                          n_slots=V).to(device)
model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Load WikiText paragraphs + tokenizer
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t for t in ds["text"] if len(t) > 200][:2000]
print(f"  paragraphs: {len(texts)}")

def make_niah_batch(bs, seq, gaps):
    x = torch.randint(V, (bs, seq))
    key_id = random.randint(2, V - 1)
    val_id = random.randint(2, V - 1)
    while val_id == key_id:
        val_id = random.randint(2, V - 1)
    gap = random.choice(gaps)
    kv_pos = random.randint(0, seq - gap - 2)
    x[:, kv_pos] = key_id
    x[:, kv_pos + 1] = val_id
    x[:, -1] = key_id
    model.slot_write(key_id, val_id)
    return x, key_id, val_id

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Fine-tune on mixed-gap NIAH (gap 32-128)
print("Fine-tuning on mixed-gap NIAH (gap=32,64,96,128)...")
for step in range(200):
    model.train()
    opt.zero_grad()
    x, _, _ = make_niah_batch(8, SEQ, [32, 64, 96, 128])
    x = x.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    if step % 50 == 49:
        print(f"  step {step+1}: loss={loss.item():.3f}")

# Evaluate on gap=64 and gap=128
print("\nNIAH evaluation after fine-tune:")
for gap in [64, 128]:
    test_x, test_y = make_niah_batch(200, SEQ, [gap])
    test_x = test_x.to(device)
    with torch.no_grad():
        logits = model(test_x)
        acc = (logits[:, -1].argmax(-1) == test_y).float().mean().item()
    print(f"  gap={gap}: recall={acc*100:.0f}%")

print("\nDone.")
