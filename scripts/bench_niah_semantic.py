"""Semantic multi-key NIAH — each model in native tokenizer space."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch, random

device = "cuda"; SEQ = 512; N_KEYS = 3
PAIRS = [("color", "blue"), ("size", "large"), ("shape", "round")]

def make_case(tok, pairs):
    """Return (input_ids, query_positions, expected_vocab_ids) for all models."""
    fill = "Although the exact origins of the tradition remain unclear, historians generally agree that the practice emerged during the late medieval period as a response to changing social and economic conditions. "
    parts = []
    kv_positions = {}
    for k, v in pairs:
        parts.append(f" {k} is {v}.")
    parts.append(f" The {pairs[0][0]} is what?")
    text = fill + " ".join(parts)
    enc = tok.encode(text)
    ids = enc.ids if hasattr(enc, "ids") else list(enc)
    ids = ids[:SEQ]
    if len(ids) < SEQ:
        pad = tok.encode(" the")
        pad_id = pad.ids[0] if hasattr(pad, "ids") else pad[0]
        ids = ids + [pad_id] * (SEQ - len(ids))
    return ids

def get_ids(tok, texts):
    enc = tok.encode(texts) if isinstance(texts, str) else tok.encode(texts[0])
    return enc.ids if hasattr(enc, "ids") else list(enc)

def score(name, model, tok, vocab, n_trials=50):
    def enc_id(t):
        e = tok.encode(t)
        return e.ids[0] if hasattr(e, "ids") else e[0]
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_trials):
            ids = make_case(tok, PAIRS)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            for k, v in PAIRS:
                total += 1
                if logits[0, -1].argmax().item() == enc_id(f" {v}"):
                    correct += 1
    return 100 * correct / max(total, 1)

print("Loading models...")

from rina import TemporalSNNModel
sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
rina = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5).to(device).eval()
rina.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)

tok4096 = __import__("tokenizers", fromlist=["Tokenizer"]).Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
tok_gpt = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tok_gpt.pad_token = tok_gpt.eos_token

from transformers import AutoModelForCausalLM, AutoTokenizer
llama = AutoModelForCausalLM.from_pretrained("D:/Software_Development/Project/models/Llama-3.2-1B", torch_dtype=torch.float16).to(device).eval()
tok_llama = AutoTokenizer.from_pretrained("D:/Software_Development/Project/models/Llama-3.2-1B")
tok_llama.pad_token = tok_llama.eos_token

print(f"\n{'Model':<25} {'Recall':>10}")
print("-" * 37)
for name, m, t, v in [("RINA 15M", rina, tok4096, 4096),
                        ("GPT-2 124M", gpt2, tok_gpt, 50257),
                        ("LLaMA 1B", llama, tok_llama, 128000)]:
    s = score(name, m, t, v)
    print(f"{name:<25} {s:>10.1f}%")
