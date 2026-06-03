"""
Seq-len benchmark: SNN v2, V1 CANN-SSM, ablation, GPT-2.
Native WikiText-103 paragraphs, no concat/pad.
Measures ppl at seq=64/128/256/512 for all models.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F

device = "cuda"; torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

V = 4096
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Collect paragraphs by length
print("Loading paragraphs...", flush=True)
paras = {64: [], 128: [], 256: [], 512: []}
for t in ds["text"][:50000]:
    if len(t) > 100:
        ids = tok.encode(t).ids
        for seq_len in [512, 256, 128, 64]:
            if len(ids) >= seq_len and len(paras[seq_len]) < 100:
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

# Build models
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
gpt2_cfg = GPT2Config(vocab_size=V, n_embd=416, n_layer=6, n_head=8, n_positions=512)
gpt2 = GPT2LMHeadModel(gpt2_cfg).to(device)
# Manual load GPT-2: copy matching keys, handle position embedding
sd = gpt2_ckpt["model"] if "model" in gpt2_ckpt else gpt2_ckpt
gpt2_sd = gpt2.state_dict()
load_count = 0
for k in gpt2_sd:
    if k in sd:
        if gpt2_sd[k].shape == sd[k].shape:
            gpt2_sd[k].copy_(sd[k])
            load_count += 1
        else:
            # Position embedding size mismatch: copy first min(n1,n2) positions
            if "wpe" in k:
                n_trained = sd[k].shape[0]
                n_model = gpt2_sd[k].shape[0]
                gpt2_sd[k][:n_trained] = sd[k][:n_trained]
                print(f"  GPT-2 wpe: extended {n_trained}→{n_model} (truncated copy)", flush=True)
                load_count += 1
gpt2.load_state_dict(gpt2_sd)
print(f"  GPT-2 loaded {load_count}/{len(gpt2_sd)} keys", flush=True)
gpt2.eval()

models = [
    ("SNN v2", snn),
    ("V1 CANN", v1),
    ("Ablation", abl),
    ("GPT-2", gpt2),
]

# Run benchmark
print("\nRunning benchmark...", flush=True)
print(f"\n{'Model':<12} {'Seq=64':>8} {'Seq=128':>8} {'Seq=256':>8} {'Seq=512':>8}")
print("-" * 48)

for name, model in models:
    results = []
    for seq_len in [64, 128, 256, 512]:
        batch = torch.stack(paras[seq_len][:30]).to(device)
        with torch.no_grad():
            out = model(batch)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
            ppl = torch.exp(loss).item()
        results.append(f"{ppl:.1f}")
    print(f"{name:<12} {'  '.join(r for r in results)}")
    print(flush=True)

print("Done.")
