"""Diffusion generation — parameter sweep for best output."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch, math
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
SEQ = 512; GEN_LEN = 80

tok = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).cuda()
sd = torch.load('checkpoints/mohe_repr_align.pt', map_location='cpu', weights_only=False)['model']
model.load_state_dict(sd, strict=False)
model.eval()

prompts = ['The meaning of life is', 'Hello world']

# ── Parameter grids ──
configs = [
    # name            steps  temp_start   topk   freeze_schedule
    ("slow_cool",      50,    1.0,         20,    "linear"),
    ("fast_cool",      20,    0.8,         50,    "linear"),
    ("step_anneal",    40,    0.9,         30,    "step"),    
    ("early_freeze",   35,    0.7,         10,    "early"),
]

for cname, N_STEPS, T_START, TOPK, freeze_t in configs:
    print(f'\n=== {cname} ({N_STEPS} steps, T={T_START}, top-{TOPK}) ===')
    for prompt in prompts:
        pids = tok.encode(prompt)
        L = len(pids)
        total_len = L + GEN_LEN

        # start from random noise (skip token 0 = EOT)
        gen = torch.randint(10, 50000, (1, total_len), device=device)
        gen[0, :L] = torch.tensor(pids, device=device)

        with torch.no_grad():
            for step in range(N_STEPS):
                # temperature: cosine anneal from T_START to 0.1
                progress = step / N_STEPS
                temp = 0.1 + (T_START - 0.1) * 0.5 * (1 + math.cos(math.pi * progress))

                inp = gen[:, -SEQ:].clone()
                out = model(inp)
                logits = (out[0] if isinstance(out, tuple) else out)[0]

                probs = torch.softmax(logits / temp, dim=-1)

                # top-k filtering: only sample from the k most probable tokens
                top_probs, top_idx = torch.topk(probs, TOPK, dim=-1)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                sampled = top_idx.gather(-1, torch.multinomial(top_probs, 1)).squeeze(-1)

                # determine which positions to resample
                if freeze_t == "linear":
                    freeze_ratio = progress  # 0 at start, 1 at end
                    keep_mask = torch.rand(probs.shape[0], device=device) < freeze_ratio
                elif freeze_t == "step":
                    keep_mask = torch.zeros(probs.shape[0], dtype=torch.bool, device=device)
                    if step > N_STEPS * 0.6:
                        keep_mask = torch.rand(probs.shape[0], device=device) < 0.3
                    if step > N_STEPS * 0.8:
                        keep_mask = torch.rand(probs.shape[0], device=device) < 0.7
                elif freeze_t == "early":
                    keep_mask = torch.zeros(probs.shape[0], dtype=torch.bool, device=device)
                    if step > N_STEPS * 0.4:
                        confidence = probs.max(dim=-1).values
                        keep_mask = confidence > 0.5 - step/N_STEPS * 0.3

                ctx = logits.shape[0]
                gen_start = max(0, total_len - SEQ)
                gen_len = min(GEN_LEN, ctx - (total_len - gen_start))

                if gen_len > 0:
                    gs = gen_start
                    ge = gs + gen_len
                    off = gs - (total_len - SEQ)
                    keep_sel = keep_mask[off:off+gen_len] if off + gen_len <= len(keep_mask) else torch.zeros(gen_len, dtype=torch.bool, device=device)
                    gen[0, gs:ge] = torch.where(keep_sel, gen[0, gs:ge], sampled[off:off+gen_len])

        text = tok.decodeBytes(gen[0, :L+60].tolist()).decode('utf-8', errors='replace')
        safe = ''.join(c if ord(c)<128 else '?' for c in text)
        print(f'  {prompt}: {safe[:200]}')
