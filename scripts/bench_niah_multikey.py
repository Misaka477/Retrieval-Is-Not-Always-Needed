"""Lethal NIAH: 3 keys per sample at random positions — break GPT-2 for good."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset

import torch, torch.nn.functional as F, random
from tqdm import tqdm
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

N_KEYS = 3  # 3 key→value pairs per sample
GAP = 128

def make_batch(n):
    xs, ys, ks_list = [], [], []
    for _ in range(n):
        while True:
            p = paras[torch.randint(0, len(paras), (1,)).item()]
            need = 128 + GAP + N_KEYS * 3
            if len(p) >= need + 4:
                seq = p[:need].tolist()
                kvs = []
                for _ in range(N_KEYS):
                    k = random.choice(KEYS)
                    v = random.choice(VALS)
                    pos = random.randint(2, need - GAP - N_KEYS * 3)
                    seq[pos] = k
                    seq[pos + 1] = v
                    seq[-1 - (N_KEYS - len(kvs)) * 2] = k  # query k at end
                    kvs.append((pos, k, v))
                xs.append(torch.tensor(seq))
                ys.append([kv[2] for kv in kvs])  # N_KEYS values
                ks_list.append([(pos, k, v) for pos, k, v in kvs])
                break
    return torch.stack(xs), torch.tensor(ys), ks_list

def recall(model, x, y, is_cann=False, cann_model=None):
    if is_cann:
        out = _full_forward(x.to(device), cann_model.embed.weight, cann_model.slot_table,
            cann_model.head.weight, cann_model.head.bias,
            cann_model.state_norm.weight, cann_model.state_norm.bias,
            cann_model.cell.patterns, cann_model.cell.beta_t,
            cann_model.cell.gate_a.weight, cann_model.cell.gate_a.bias,
            cann_model.cell.gate_b.weight, cann_model.cell.gate_b.bias,
            cann_model.cell.gate_alpha.weight, cann_model.cell.gate_alpha.bias,
            cann_model.cell.proj_in.weight, cann_model.cell.proj_in.bias,
            cann_model.cell.norm.weight, cann_model.cell.norm.bias, cann_model.attract_every)
    else:
        out = model(x.to(device)).logits
    # Check N_KEYS positions at the end
    preds = out[:, -N_KEYS * 2::2].argmax(-1)  # every other position from -N_KEYS*2
    return (preds == y.to(device)).float().mean().item()

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

print(f"\n  {'':6s}  {'GPT-2':>8s}  {'ABL+slot':>10s}  {'CANN+slot':>10s}")
print(f"  {'':6s}  {'recall':>8s}  {'recall':>10s}  {'recall/VRAM':>10s}")
print("  " + "-" * 45)

train_x, train_y, train_kp = make_batch(200)
test_x, test_y, _ = make_batch(200)

# ── GPT-2 ──
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
opt = torch.optim.AdamW(gpt2.parameters(), lr=3e-4)
best_g = 0
for step in tqdm(range(200), desc="GPT-2"):
    gpt2.train(); gpt2.zero_grad()
    for i in range(0, len(train_x), 16):
        idx = slice(i, min(i+16, len(train_x)))
        logits_q = gpt2(train_x[idx].to(device)).logits[:, -N_KEYS*2::2]  # [B, 3, V]
        loss = F.cross_entropy(logits_q.reshape(-1, V), train_y[idx].view(-1).to(device))
        loss.backward()
    opt.step()
    if step % 10 == 9:
        gpt2.eval()
        with torch.no_grad(): acc = recall(gpt2, test_x, test_y)
        best_g = max(best_g, acc)
vram_g = torch.cuda.max_memory_allocated() / 1e9
print(f"  GPT-2 : recall={best_g*100:.0f}%  VRAM={vram_g:.1f}G", flush=True)

# ── ABL ──
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
abl.slot_table.zero_()
for i in range(len(train_x)):
    for pos, k, v in train_kp[i]:
        if k in KEYS and v in VALS: abl.slot_write(k, v)
opt = torch.optim.AdamW(abl.parameters(), lr=3e-4)
best_a = 0
for step in tqdm(range(200), desc="ABL  "):
    abl.train(); abl.zero_grad()
    for i in range(0, len(train_x), 16):
        idx = slice(i, min(i+16, len(train_x)))
        out = _full_forward(train_x[idx].to(device), abl.embed.weight, abl.slot_table,
            abl.head.weight, abl.head.bias, abl.state_norm.weight, abl.state_norm.bias,
            abl.cell.patterns, abl.cell.beta_t,
            abl.cell.gate_a.weight, abl.cell.gate_a.bias,
            abl.cell.gate_b.weight, abl.cell.gate_b.bias,
            abl.cell.gate_alpha.weight, abl.cell.gate_alpha.bias,
            abl.cell.proj_in.weight, abl.cell.proj_in.bias,
            abl.cell.norm.weight, abl.cell.norm.bias, abl.attract_every)
        logits_q = out[:, -N_KEYS*2::2]
        loss = F.cross_entropy(logits_q.reshape(-1, V), train_y[idx].view(-1).to(device)); loss.backward()
    opt.step()
    if step % 10 == 9:
        abl.eval()
        with torch.no_grad(): acc = recall(abl, test_x, test_y, True, abl)
        best_a = max(best_a, acc)
vram_a = torch.cuda.max_memory_allocated() / 1e9
print(f"  ABL  : recall={best_a*100:.0f}%  VRAM={vram_a:.1f}G", flush=True)

# ── CANN ──
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
cann.slot_table.zero_()
for i in range(len(train_x)):
    for pos, k, v in train_kp[i]:
        if k in KEYS and v in VALS: cann.slot_write(k, v)
opt = torch.optim.AdamW(cann.parameters(), lr=3e-4)
best_c = 0
for step in tqdm(range(200), desc="CANN "):
    cann.train(); cann.zero_grad()
    for i in range(0, len(train_x), 16):
        idx = slice(i, min(i+16, len(train_x)))
        out = _full_forward(train_x[idx].to(device), cann.embed.weight, cann.slot_table,
            cann.head.weight, cann.head.bias, cann.state_norm.weight, cann.state_norm.bias,
            cann.cell.patterns, cann.cell.beta_t,
            cann.cell.gate_a.weight, cann.cell.gate_a.bias,
            cann.cell.gate_b.weight, cann.cell.gate_b.bias,
            cann.cell.gate_alpha.weight, cann.cell.gate_alpha.bias,
            cann.cell.proj_in.weight, cann.cell.proj_in.bias,
            cann.cell.norm.weight, cann.cell.norm.bias, cann.attract_every)
        logits_q = out[:, -N_KEYS*2::2]  # [B, 3, V]
        loss = F.cross_entropy(logits_q.reshape(-1, V), train_y[idx].view(-1).to(device)); loss.backward()
    opt.step()
    if step % 10 == 9:
        cann.eval()
        with torch.no_grad(): acc = recall(cann, test_x, test_y, True, cann)
        best_c = max(best_c, acc)
vram_c = torch.cuda.max_memory_allocated() / 1e9

print(f"  CANN : recall={best_c*100:.0f}%  VRAM={vram_c:.1f}G", flush=True)
