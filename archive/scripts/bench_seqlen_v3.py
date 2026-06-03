"""
Seq-len benchmark v3: SNN v2, V1 CANN-SSM, ablation, GPT-2.
Native paragraphs, 3 runs averaged. Adds seq=1024/2048 where possible.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F, random

device = "cuda"; torch.manual_seed(42); random.seed(42)
V = 4096
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Collect paragraphs by length (wider search for long ones)
print("Loading paragraphs...", flush=True)
seq_lens = [64, 128, 256, 512, 1024, 2048]
paras = {s: [] for s in seq_lens}
for t in ds["text"]:
    if len(t) > 100:
        ids = tok.encode(t).ids
        for seq_len in reversed(seq_lens):
            if len(ids) >= seq_len and len(paras[seq_len]) < 300:
                paras[seq_len].append(torch.tensor(ids[:seq_len], dtype=torch.long))
                break

for k, v in paras.items():
    print(f"  seq={k}: {len(v)} paragraphs", flush=True)

def load_model(ckpt_path, model_cls, **kwargs):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = model_cls(V, **kwargs).to(device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(sd, strict=False)
    return m.eval()

print("\nLoading models...", flush=True)
from rina import TemporalSNNModel
snn = load_model("checkpoints/cann_snn15m_v2_ep12.pt", TemporalSNNModel,
                 d_model=840, n_patterns=4096, beta=0.5, attract_every=2,
                 error_threshold=1.0, hebbian_lr=0.0, inhibition_threshold=0.0)

from modules.cann_ssm import RINASeqModel
v1 = load_model("checkpoints/cann_15m_wt103_final.pt", RINASeqModel,
                d_model=768, n_patterns=4096, n_slots=0, attract_every=2)
abl = load_model("checkpoints/cann_15m_abl_final.pt", RINASeqModel,
                 d_model=768, n_patterns=4096, n_slots=0, attract_every=9999)

from transformers import GPT2Config, GPT2LMHeadModel
gpt2_ckpt = torch.load("checkpoints/gpt2_15m_wt103_final.pt", map_location=device, weights_only=False)
gpt2_cfg = GPT2Config(vocab_size=V, n_embd=416, n_layer=6, n_head=8, n_positions=2048)
gpt2 = GPT2LMHeadModel(gpt2_cfg).to(device)
sd = gpt2_ckpt["model"] if "model" in gpt2_ckpt else gpt2_ckpt
gpt2_sd = gpt2.state_dict()
load_count = 0
for k in gpt2_sd:
    if k in sd:
        if gpt2_sd[k].shape == sd[k].shape:
            gpt2_sd[k].copy_(sd[k])
            load_count += 1
        elif "wpe" in k:
            n_t = sd[k].shape[0]
            gpt2_sd[k][:n_t] = sd[k][:n_t]
            print(f"  GPT-2 wpe: {n_t}->{gpt2_sd[k].shape[0]}", flush=True)
            load_count += 1
gpt2.load_state_dict(gpt2_sd)
print(f"  GPT-2 loaded {load_count}/{len(gpt2_sd)}", flush=True)
gpt2.eval()

models = [
    ("SNN v2", snn),
    ("V1 CANN", v1),
    ("Ablation", abl),
    ("GPT-2", gpt2),
]

# Run benchmark: 3 trials, sample 30 paragraphs each time
print("\nRunning benchmark (3 trials averaged)...", flush=True)
N_TRIALS, N_SAMPLES = 3, 30
header = f"{'Model':<12}"
for s in seq_lens:
    header += f"  {'Seq='+str(s):>10}"
print(header)
print("-" * (12 + 11 * len(seq_lens)))

for name, model in models:
    row = f"{name:<12}"
    for seq_len in seq_lens:
        pool = paras[seq_len]
        if len(pool) < N_SAMPLES:
            row += f"  {'N/A':>10}"
            continue
        ppls = []
        for trial in range(N_TRIALS):
            idx = random.sample(range(len(pool)), N_SAMPLES)
            batch = torch.stack([pool[i] for i in idx]).to(device)
            with torch.no_grad():
                out = model(batch)
                logits = out.logits if hasattr(out, "logits") else out
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
                ppls.append(torch.exp(loss).item())
        mean = sum(ppls) / len(ppls)
        sd_val = (sum((p - mean)**2 for p in ppls) / len(ppls))**0.5
        row += f"  {mean:>5.1f}\u00b1{sd_val:.1f}"
    print(row)
    print(flush=True)

print("Done.")
