"""
WikiText-103 验证集 ppl 对比 — 各模型用各的 tokenizer。
RINA: BPE 4096 | GPT-2: Native 50K | LLaMA: Native 128K
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from rina import TemporalSNNModel
import torch, torch.nn.functional as F

device = "cuda"; SEQ, BS = 64, 8

# Load validation data (raw text)
ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:200]

def get_ppl(encode_fn, model, vocab, name, max_len=640):
    ids_list = []
    for t in texts:
        enc = encode_fn(t, max_len)
        if len(enc) >= 64:
            ids_list.append(torch.tensor(enc, dtype=torch.long))
    ids = torch.cat(ids_list) if ids_list else torch.zeros(0, dtype=torch.long)
    nb = max(1, (len(ids) - 1) // (BS * SEQ))
    model.eval(); tl = 0.0
    with torch.no_grad():
        for bi in range(nb):
            x = ids[bi * BS * SEQ : bi * BS * SEQ + BS * SEQ].view(BS, SEQ).to(device)
            o = model(x); lo = o.logits if hasattr(o, "logits") else o
            tl += F.cross_entropy(lo[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).item()
    p = torch.exp(torch.tensor(tl / nb)).item()
    print(f"  {name:<25} {p:.1f}")
    return p

print(f"{'Model':<30} {'WikiText valid ppl':>20}")
print("-" * 50)

# RINA — BPE 4096
tok = __import__("tokenizers", fromlist=["Tokenizer"]).Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
def enc_rina(t, max_len): return tok.encode(t).ids[:max_len]

sd = torch.load("checkpoints/cann_snn15m_v2_slot_ep13.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
get_ppl(enc_rina, m, 4096, "RINA slot")

# GPT-2 124M — native 50K
tok_gpt = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tok_gpt.pad_token = tok_gpt.eos_token
def enc_gpt(t, max_len): return tok_gpt.encode(t, max_length=max_len, truncation=True)

g = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
get_ppl(enc_gpt, g, g.config.vocab_size, "GPT-2 124M")

# LLaMA 3.2 1B — native 128K
tok_llama = AutoTokenizer.from_pretrained("D:/Software_Development/Project/models/Llama-3.2-1B")
tok_llama.pad_token = tok_llama.eos_token
def enc_llama(t, max_len): return tok_llama.encode(t, max_length=max_len, truncation=True)

llama = AutoModelForCausalLM.from_pretrained("D:/Software_Development/Project/models/Llama-3.2-1B", torch_dtype=torch.float16).to(device).eval()
get_ppl(enc_llama, llama, llama.config.vocab_size, "LLaMA 3.2 1B")

