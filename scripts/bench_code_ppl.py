"""RINA code ppl comparison.
Usage:
  python scripts/bench_code_ppl.py --seq 128
  python scripts/bench_code_ppl.py --seq 256 --th 1.0,0.5,0.3
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

device = "cuda"
SEQ = int(sys.argv[sys.argv.index("--seq") + 1]) if "--seq" in sys.argv else 64
BS = 4 if SEQ >= 128 else 8
TH_VALS = [float(x) for x in sys.argv[sys.argv.index("--th") + 1].split(",")] if "--th" in sys.argv else [1.0]
V_RINA = 4096; V_GPT = 4096

# Load code data
print("Loading StarCoder Python code...", flush=True)
ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
code_texts = []
for i, sample in enumerate(ds):
    if i >= 200: break
    code_texts.append(sample["content"])

# Tokenize
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
all_ids = []
for t in tqdm(code_texts, desc="tokenizing"):
    ids = tok.encode(t).ids
    if len(ids) >= SEQ:
        all_ids.append(torch.tensor(ids[:min(len(ids), SEQ * 100)], dtype=torch.long))
ids = torch.cat(all_ids) if all_ids else torch.zeros(0, dtype=torch.long)
num_batches = max(1, (len(ids) - 1) // (BS * SEQ))
print(f"  tokens: {len(ids):,}, batches: {num_batches}")

results = []

# Helper: build model, load checkpoint, eval at given threshold
def eval_checkpoint(ckpt_path, name, th):
    if not os.path.exists(ckpt_path):
        results.append((name, -1))
        return
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = TemporalSNNModel(V_RINA, d_model=840, n_patterns=4096, beta=0.5,
                         attract_every=2, error_threshold=th,
                         hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.eval()
    total_loss = 0.0
    with torch.no_grad():
        for bi in tqdm(range(num_batches), desc=f"{name[:16]} th={th}", leave=False):
            start = bi * BS * SEQ
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_RINA), x[:, 1:].reshape(-1))
            total_loss += loss.item()
    ppl = torch.exp(torch.tensor(total_loss / num_batches)).item()
    results.append((f"RINA (th={th})", ppl))
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

# 1. Load code checkpoint and eval at each threshold
ckpt_path = "checkpoints/code_seq256_resume.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "checkpoints/code_seq128_resume.pt"
print(f"Checkpoint: {ckpt_path}", flush=True)
for th in TH_VALS:
    eval_checkpoint(ckpt_path, "RINA", th)

# 2. RINA pretrain only (skip if --seq requested)
if "--seq" not in sys.argv:
    import gc; gc.collect(); torch.cuda.empty_cache()
    if os.path.exists("checkpoints/cann_snn15m_v2_slot_ep12.pt"):
        eval_checkpoint("checkpoints/cann_snn15m_v2_slot_ep12.pt", "slot-pretrain", 1.0)

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
    results.append(("GPT-2 15M", ppl))

print(f"\n{'Model':<20} {'ppl':>8}")
print("-" * 30)
for name, ppl in results:
    print(f"{name:<20} {ppl:>8.2f}")

