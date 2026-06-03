"""Seq-len benchmark v2: native paragraphs, no concatenation, no pad."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from modules.cann_ssm import RINASeqModel, _full_forward
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda"; torch.manual_seed(42)
DM, NP = 768, 4096
CKPT_CANN = "checkpoints/cann_15m_wt103_final.pt"
CKPT_ABL  = "checkpoints/cann_15m_abl_final.pt"
CKPT_GPT2 = "checkpoints/gpt2_15m_wt103_final.pt"

print("Loading data...", flush=True)
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
V = tok.get_vocab_size()

# Tokenize paragraphs — use subset for speed
all_paragraphs = []
for t in ds["text"][:50000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 64:
            all_paragraphs.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(all_paragraphs)}", flush=True)

print("Loading models...", flush=True)
cann = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=2).to(device)
cann.load_state_dict(torch.load(CKPT_CANN, map_location=device)); cann.eval()

abl = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=9999).to(device)
abl.load_state_dict(torch.load(CKPT_ABL, map_location=device)); abl.eval()

cfg = GPT2Config(vocab_size=V, n_embd=416, n_layer=6, n_head=8, n_positions=512)
gpt2 = GPT2LMHeadModel(cfg).to(device)
state = torch.load(CKPT_GPT2, map_location=device)
wpe_old = state["transformer.wpe.weight"]
wpe_new = torch.zeros(512, 416, device=state["transformer.wpe.weight"].device)
wpe_new[:64] = wpe_old; wpe_new[64:] = wpe_old[-1:].repeat(448, 1)
state["transformer.wpe.weight"] = wpe_new
gpt2.load_state_dict(state); gpt2.eval()

@torch.no_grad()
def ppl_cann(model, x, y):
    with torch.autocast("cuda", dtype=torch.float16):
        out = _full_forward(x, model.embed.weight, model.slot_table,
            model.head.weight, model.head.bias, model.state_norm.weight, model.state_norm.bias,
            model.cell.patterns, model.cell.beta_t,
            model.cell.gate_a.weight, model.cell.gate_a.bias,
            model.cell.gate_b.weight, model.cell.gate_b.bias,
            model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
            model.cell.proj_in.weight, model.cell.proj_in.bias,
            model.cell.norm.weight, model.cell.norm.bias, model.attract_every)
    return F.cross_entropy(out.view(-1, V), y.view(-1)).item()

@torch.no_grad()
def ppl_gpt2(model, x, y):
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x).logits
    return F.cross_entropy(out.view(-1, V), y.view(-1)).item()

@torch.no_grad()
def measure(model, seq_len, fn, n_samples=50):
    losses = []
    for _ in range(n_samples):
        # Pick a paragraph long enough
        while True:
            p = all_paragraphs[torch.randint(0, len(all_paragraphs), (1,)).item()]
            if len(p) > seq_len:
                break
        start = torch.randint(0, len(p) - seq_len, (1,)).item()
        x = p[start:start+seq_len].unsqueeze(0).to(device)
        y = p[start+1:start+seq_len+1].unsqueeze(0).to(device)
        try:
            losses.append(fn(model, x, y))
        except torch.cuda.OutOfMemoryError:
            return float('nan')
    return torch.exp(torch.tensor(sum(losses) / len(losses))).item()

print(f"\n  {'seq':>4s}  {'CANN':>7s}  {'ABL':>7s}  {'GPT-2':>7s}  |  {'delta_abl':>9s}  {'delta_gpt2':>9s}")
print("  " + "-" * 55)
for seq, n in [(64, 50), (128, 50), (256, 50), (512, 30)]:
    pc = measure(cann, seq, ppl_cann, n)
    pa = measure(abl,  seq, ppl_cann, n)
    pg = measure(gpt2, seq, ppl_gpt2, n)
    print(f"  {seq:4d}  {pc:7.1f}  {pa:7.1f}  {pg:7.1f}  |  {pa-pc:+9.1f}  {pg-pc:+9.1f}")
torch.cuda.empty_cache()
print()

# Also check how many paragraphs are long enough
for l in [64, 128, 256, 512]:
    cnt = sum(1 for p in all_paragraphs if len(p) > l)
    print(f"  paragraphs > {l:3d} tokens: {cnt}")
