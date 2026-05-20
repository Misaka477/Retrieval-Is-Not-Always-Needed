"""Extreme NIAH: random key position — breaks GPT-2's fixed-offset cheat."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset

import torch, torch.nn.functional as F, random
from modules.cann_ssm import RINASeqModel, _full_forward
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda"; torch.manual_seed(42); random.seed(42)

V = 4096; DM = 768; NP = 4096
KEYS = list(range(1, 6)); VALS = list(range(6, 11))
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
paras = []
for t in ds["text"][:10000]:
    if len(t) > 100:
        enc = tok.encode(t).ids
        if len(enc) >= 200:
            paras.append(torch.tensor(enc, dtype=torch.long))
print(f"  paragraphs: {len(paras)}", flush=True)

def make_batch(n, gap):
    xs, ys, ks = [], [], []
    for _ in range(n):
        while True:
            p = paras[torch.randint(0, len(paras), (1,)).item()]
            need = 64 + gap + 4
            if len(p) >= need + 4:
                k = random.choice(KEYS); v = random.choice(VALS)
                ins = random.randint(2, min(need - 4, len(p) - need))
                seq = p[:need].tolist()
                seq[ins] = k; seq[ins+1] = v; seq[-1] = k
                xs.append(torch.tensor(seq))
                ys.append(v)
                ks.append((ins, k))  # track position for slot
                break
    return torch.stack(xs), torch.tensor(ys), ks

# ── Load models ──
cann = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, n_slots=V, attract_every=2).to(device)
cann.load_state_dict(torch.load("checkpoints/cann_15m_wt103_final.pt", map_location=device))
abl = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5, n_slots=V, attract_every=9999).to(device)
abl.load_state_dict(torch.load("checkpoints/cann_15m_abl_final.pt", map_location=device))

cfg = GPT2Config(vocab_size=V, n_embd=416, n_layer=6, n_head=8, n_positions=512)
gpt2 = GPT2LMHeadModel(cfg).to(device)
st = torch.load("checkpoints/gpt2_15m_wt103_final.pt", map_location=device)
wpe_old = st["transformer.wpe.weight"]
wpe_new = torch.zeros(512, 416, device=wpe_old.device)
wpe_new[:64] = wpe_old; wpe_new[64:] = wpe_old[-1:].repeat(448, 1)
st["transformer.wpe.weight"] = wpe_new
gpt2.load_state_dict(st)

print("\n  gap    GPT-2     ABL+slot  CANN+slot")
print("  ──────────────────────────────────────────")

for gap in [128]:
    train_x, train_y, train_kp = make_batch(200, gap)
    test_x, test_y, _ = make_batch(200, gap)

    # GPT-2 (fastest)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(gpt2.parameters(), lr=3e-4)
    best_g = 0
    for step in range(200):
        gpt2.train(); gpt2.zero_grad()
        for i in range(0, len(train_x), 32):
            idx = slice(i, min(i+32, len(train_x)))
            loss = F.cross_entropy(gpt2(train_x[idx].to(device)).logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()
        if step % 10 == 9:
            gpt2.eval()
            with torch.no_grad():
                acc = (gpt2(test_x.to(device)).logits[:, -1].argmax(-1)==test_y.to(device)).float().mean().item()
            best_g = max(best_g, acc)
    vram_g = torch.cuda.max_memory_allocated() / 1e9

    # ABL (medium)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    abl.slot_table.zero_()
    opt = torch.optim.AdamW(abl.parameters(), lr=3e-4)
    best_a = 0
    for step in range(200):
        abl.train(); abl.zero_grad()
        for i in range(0, len(train_x), 32):
            idx = slice(i, min(i+32, len(train_x)))
            out = _full_forward(train_x[idx].to(device), abl.embed.weight, abl.slot_table,
                abl.head.weight, abl.head.bias, abl.state_norm.weight, abl.state_norm.bias,
                abl.cell.patterns, abl.cell.beta_t,
                abl.cell.gate_a.weight, abl.cell.gate_a.bias,
                abl.cell.gate_b.weight, abl.cell.gate_b.bias,
                abl.cell.gate_alpha.weight, abl.cell.gate_alpha.bias,
                abl.cell.proj_in.weight, abl.cell.proj_in.bias,
                abl.cell.norm.weight, abl.cell.norm.bias, abl.attract_every)
            loss = F.cross_entropy(out[:, -1], train_y[idx].to(device)); loss.backward()
        opt.step()
        for i in range(len(train_x)):
            ins, k = train_kp[i]; v = int(train_y[i])
            if k in KEYS and v in VALS: abl.slot_write(k, v)
        if step % 10 == 9:
            abl.eval()
            with torch.no_grad():
                out = _full_forward(test_x.to(device), abl.embed.weight, abl.slot_table,
                    abl.head.weight, abl.head.bias, abl.state_norm.weight, abl.state_norm.bias,
                    abl.cell.patterns, abl.cell.beta_t,
                    abl.cell.gate_a.weight, abl.cell.gate_a.bias,
                    abl.cell.gate_b.weight, abl.cell.gate_b.bias,
                    abl.cell.gate_alpha.weight, abl.cell.gate_alpha.bias,
                    abl.cell.proj_in.weight, abl.cell.proj_in.bias,
                    abl.cell.norm.weight, abl.cell.norm.bias, abl.attract_every)
                acc = (out[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best_a = max(best_a, acc)
    vram_a = torch.cuda.max_memory_allocated() / 1e9

    # CANN (slowest)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cann.slot_table.zero_()
    opt = torch.optim.AdamW(cann.parameters(), lr=3e-4)
    best_c = 0
    for step in range(200):
        cann.train(); cann.zero_grad()
        for i in range(0, len(train_x), 32):
            idx = slice(i, min(i+32, len(train_x)))
            out = _full_forward(train_x[idx].to(device), cann.embed.weight, cann.slot_table,
                cann.head.weight, cann.head.bias, cann.state_norm.weight, cann.state_norm.bias,
                cann.cell.patterns, cann.cell.beta_t,
                cann.cell.gate_a.weight, cann.cell.gate_a.bias,
                cann.cell.gate_b.weight, cann.cell.gate_b.bias,
                cann.cell.gate_alpha.weight, cann.cell.gate_alpha.bias,
                cann.cell.proj_in.weight, cann.cell.proj_in.bias,
                cann.cell.norm.weight, cann.cell.norm.bias, cann.attract_every)
            loss = F.cross_entropy(out[:, -1], train_y[idx].to(device)); loss.backward()
        opt.step()
        for i in range(len(train_x)):
            ins, k = train_kp[i]; v = int(train_y[i])
            if k in KEYS and v in VALS: cann.slot_write(k, v)
        if step % 10 == 9:
            cann.eval()
            with torch.no_grad():
                out = _full_forward(test_x.to(device), cann.embed.weight, cann.slot_table,
                    cann.head.weight, cann.head.bias, cann.state_norm.weight, cann.state_norm.bias,
                    cann.cell.patterns, cann.cell.beta_t,
                    cann.cell.gate_a.weight, cann.cell.gate_a.bias,
                    cann.cell.gate_b.weight, cann.cell.gate_b.bias,
                    cann.cell.gate_alpha.weight, cann.cell.gate_alpha.bias,
                    cann.cell.proj_in.weight, cann.cell.proj_in.bias,
                    cann.cell.norm.weight, cann.cell.norm.bias, cann.attract_every)
                acc = (out[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best_c = max(best_c, acc)
    vram_c = torch.cuda.max_memory_allocated() / 1e9

    print(f"  {gap:3d}     {best_g*100:.0f}%/{vram_g:.1f}G  {best_a*100:.0f}%/{vram_a:.1f}G   {best_c*100:.0f}%/{vram_c:.1f}G")
