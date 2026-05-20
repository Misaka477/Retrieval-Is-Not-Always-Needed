"""Autoregressive generation demo for SNN v2."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_DATASETS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch, torch.nn.functional as F
from tokenizers import Tokenizer
from modules.temporal_snn_cell import TemporalSNNModel

device = "cuda"

CKPT = "checkpoints/cann_snn15m_v2_ep12.pt"
tok = Tokenizer.from_file("checkpoints/cann_snn15m_v2")

print("Loading model...", flush=True)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = TemporalSNNModel(tok.get_vocab_size(), d_model=840, n_patterns=4096, beta=0.5,
                          attract_every=2, error_threshold=1.0, hebbian_lr=0.0,
                          inhibition_threshold=0.0).to(device).eval()
model.load_state_dict(ckpt["model"], strict=False)
print("  Done.", flush=True)

def generate(prompt, max_len=128, temperature=0.8, top_k=20):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(x)
        logit = logits[0, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logit, top_k)
            logit[logit < v[-1]] = float("-inf")
        probs = F.softmax(logit, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        x = torch.cat([x[:, -63:], torch.tensor([[next_id]], device=device)], dim=1)
        yield next_id

prompts = [
    "The meaning of life is",
    "In the beginning, the universe was",
    "The most important scientific discovery of the",
]

for prompt in prompts:
    print(f"\nP: {prompt}")
    print("G: ", end="", flush=True)
    ids = tok.encode(prompt).ids
    gen_ids = list(ids)
    for token in generate(prompt, max_len=64, temperature=0.7, top_k=10):
        gen_ids.append(token)
        try:
            txt = tok.decode(gen_ids)[len(prompt):]
            print(txt.encode('gbk', errors='replace').decode('gbk'), end='', flush=True)
        except: pass
    print()

print("\nDone.")
