"""V5 Phase 3 — comprehensive generation test (all methods)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch, math
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
SEQ = 512; GEN_LEN = 60
tok = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

print("Loading REPR-ALIGN checkpoint...")
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).cuda()
sd = torch.load('checkpoints/mohe_repr_align.pt', map_location='cpu', weights_only=False)['model']
model.load_state_dict(sd, strict=False)
model.eval()
print("Ready.\n")

prompts = ['The meaning of life is', 'Hello world', 'def fibonacci(n):']

# ── 1. Autoregressive: greedy ──
print("=== 1. Autoregressive (greedy) ===")
for p in prompts:
    gen = tok.encode(p)
    with torch.no_grad():
        for _ in range(GEN_LEN):
            out = model(torch.tensor([gen[-SEQ:]], device=device))
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :]
            tid = torch.argmax(logits).item()
            if tid == 0: break
            gen.append(tid)
    text = ''.join(c if ord(c) < 128 else '?' for c in tok.decodeBytes(gen).decode('utf-8', errors='replace'))
    print(f'  {p}: {text[:200]}\n')

# ── 2. Autoregressive: top-p sampling ──
print("=== 2. Autoregressive (top-p=0.7, temp=0.8) ===")
for p in prompts:
    gen = tok.encode(p)
    with torch.no_grad():
        for _ in range(GEN_LEN):
            out = model(torch.tensor([gen[-SEQ:]], device=device))
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :]
            logits = logits / 0.8
            probs = torch.softmax(logits, dim=-1)
            sorted_p, idx = probs.sort(descending=True)
            cumsum = sorted_p.cumsum(dim=-1)
            mask = cumsum - sorted_p > 0.7
            probs[mask] = 0
            probs = probs / probs.sum()
            tid = torch.multinomial(probs, 1).item()
            if tid == 0: break
            gen.append(tid)
    text = ''.join(c if ord(c) < 128 else '?' for c in tok.decodeBytes(gen).decode('utf-8', errors='replace'))
    print(f'  {p}: {text[:200]}\n')

# ── 3. Diffusion: iterative refinement ──
print("=== 3. Diffusion (30 steps, top-20, temp anneal) ===")
for p in prompts:
    pids = tok.encode(p)
    L = len(pids)
    total_len = L + GEN_LEN
    gen = torch.randint(10, 50000, (1, total_len), device=device)
    gen[0, :L] = torch.tensor(pids, device=device)

    with torch.no_grad():
        for step in range(30):
            temp = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / 30))
            inp = gen[:, -SEQ:].clone()
            out = model(inp)
            logits = (out[0] if isinstance(out, tuple) else out)[0]

            probs = torch.softmax(logits / temp, dim=-1)
            top_probs, top_idx = torch.topk(probs, 20, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            sampled = top_idx.gather(-1, torch.multinomial(top_probs, 1)).squeeze(-1)

            ctx = logits.shape[0]
            gen_start = L  # first generated token position
            out_start = max(0, gen_start - (total_len - SEQ))
            out_len = min(GEN_LEN, ctx - out_start, total_len - gen_start)
            if out_len > 0:
                gen[0, gen_start:gen_start+out_len] = sampled[out_start:out_start+out_len]

    text = ''.join(c if ord(c) < 128 else '?' for c in tok.decodeBytes(gen[0, :L+40].tolist()).decode('utf-8', errors='replace'))
    print(f'  {p}: {text[:200]}\n')

# ── 4. Diffusion: slow anneal (more steps) ──
print("=== 4. Diffusion (80 steps, top-10, slower cool) ===")
for p in prompts:
    pids = tok.encode(p)
    L = len(pids)
    total_len = L + GEN_LEN
    gen = torch.randint(10, 50000, (1, total_len), device=device)
    gen[0, :L] = torch.tensor(pids, device=device)

    with torch.no_grad():
        for step in range(80):
            temp = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / 80))
            inp = gen[:, -SEQ:].clone()
            out = model(inp)
            logits = (out[0] if isinstance(out, tuple) else out)[0]

            probs = torch.softmax(logits / temp, dim=-1)
            top_probs, top_idx = torch.topk(probs, 10, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            sampled = top_idx.gather(-1, torch.multinomial(top_probs, 1)).squeeze(-1)

            ctx = logits.shape[0]
            gen_start = L
            out_start = max(0, gen_start - (total_len - SEQ))
            out_len = min(GEN_LEN, ctx - out_start, total_len - gen_start)
            if out_len > 0:
                gen[0, gen_start:gen_start+out_len] = sampled[out_start:out_start+out_len]

    text = ''.join(c if ord(c) < 128 else '?' for c in tok.decodeBytes(gen[0, :L+40].tolist()).decode('utf-8', errors='replace'))
    print(f'  {p}: {text[:200]}\n')

print("Done.")
