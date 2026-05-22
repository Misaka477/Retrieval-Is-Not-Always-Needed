"""RINA vs GPT-2 15M on code — direct ppl comparison."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

device = "cuda"
SEQ, BS = 64, 8
V_RINA = 4096  # RINA vocab
V_GPT = 4096   # GPT-2 also uses 4096

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

# Load code data (same as RINA's training)
print("Loading StarCoder Python code...", flush=True)
ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
code_texts = []
for i, sample in enumerate(ds):
    if i >= 200: break
    code_texts.append(sample["content"])

# Tokenize
all_ids = []
for t in tqdm(code_texts, desc="tokenizing"):
    ids = tok.encode(t).ids
    if len(ids) >= SEQ:
        all_ids.append(torch.tensor(ids[:min(len(ids), SEQ * 100)], dtype=torch.long))
ids = torch.cat(all_ids) if all_ids else torch.zeros(0, dtype=torch.long)
num_batches = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  tokens: {len(ids):,}, batches: {num_batches}")

results = []

# 1. RINA checkpoints
checkpoints = [
    ("RINA (code-trained, ~8400st)", "checkpoints/rine_code_ep1_latest.pt"),
    ("RINA (FineWeb, no code)", "checkpoints/fineweb_resume.pt"),
]
for name, path in checkpoints:
    if not os.path.exists(path):
        results.append((name, -1))
        continue
    sd = torch.load(path, map_location=device, weights_only=False)
    m = TemporalSNNModel(V_RINA, d_model=840, n_patterns=4096, beta=0.5,
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
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_RINA), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
    results.append((name, ppl))
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

# 2. RINA pretrain only
import gc; gc.collect(); torch.cuda.empty_cache()
if os.path.exists("checkpoints/cann_snn15m_v2_slot_ep12.pt"):
    sd = torch.load("checkpoints/cann_snn15m_v2_slot_ep12.pt", map_location=device, weights_only=False)
    m = TemporalSNNModel(V_RINA, d_model=840, n_patterns=4096, beta=0.5,
                          attract_every=2, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.eval()
    total_loss = 0.0
    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc="RINA slot-pretrain", leave=False):
            start = bi * BS * SEQ
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_RINA), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
    results.append(("RINA (slot-pretrain, no code)", ppl))
    del m

# 3. GPT-2 15M
import gc; gc.collect(); torch.cuda.empty_cache()
gpt2 = GPT2LMHeadModel(GPT2Config(vocab_size=V_GPT, n_embd=416, n_layer=6, n_head=8, n_positions=1024)).to(device)
gpt2_ckpt = torch.load("checkpoints/gpt2_15m_wt103_final.pt", map_location=device, weights_only=False)
sd = gpt2_ckpt["model"] if "model" in gpt2_ckpt else gpt2_ckpt
gpt2_sd = gpt2.state_dict()
for k in gpt2_sd:
    if k in sd and gpt2_sd[k].shape == sd[k].shape:
        gpt2_sd[k].copy_(sd[k])
gpt2.load_state_dict(gpt2_sd)
gpt2.eval()
total_loss = 0.0
with torch.no_grad():
    for bi in tqdm(range(num_batches), desc="GPT-2", leave=False):
        start = bi * BS * SEQ
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        out = gpt2(x)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_GPT), x[:, 1:].reshape(-1))
        total_loss += loss.item()
ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
results.append(("GPT-2 15M (WikiText baseline)", ppl))

print(f"\n{'Model':<40} {'ppl':>8}")
print("-" * 50)
for name, ppl in results:
    print(f"{name:<40} {ppl:>8.2f}")
