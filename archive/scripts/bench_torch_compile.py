"""Benchmark torch.compile for RINA training."""
import torch
import torch.nn.functional as F
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, _cell_full, _cell_ssm

torch.set_float32_matmul_precision('high')

device = "cuda"
torch.manual_seed(42)

vocab_size = 1024
dm = 256
np_ = 4096
seq = 64
bs = 8
ae = 4

model = RINASeqModel(vocab_size, d_model=dm, n_patterns=np_,
                     beta=0.5, n_slots=vocab_size, attract_every=ae).to(device)
model.train()

x = torch.randint(0, vocab_size, (bs, seq), device=device)
tgt = torch.randint(0, vocab_size, (bs, seq), device=device)

n_warmup = 5
n_iter = 20

# Convenience accessors
pat = model.cell.patterns
beta_t = model.cell.beta_t
wa = model.cell.gate_a.weight; ba = model.cell.gate_a.bias
wb = model.cell.gate_b.weight; bb = model.cell.gate_b.bias
wg = model.cell.gate_alpha.weight; bg = model.cell.gate_alpha.bias
wp = model.cell.proj_in.weight; bp = model.cell.proj_in.bias
wn = model.cell.norm.weight; bn = model.cell.norm.bias
snw = model.state_norm.weight; snb = model.state_norm.bias
hw = model.head.weight; hb = model.head.bias
slot = model.slot_table

def bench(name, fn, x, tgt):
    for _ in range(n_warmup):
        model.zero_grad()
        out = fn(x)
        loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model.zero_grad()
        out = fn(x)
        loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t = (time.time() - t0) / n_iter * 1000
    print(f"  {name:35s} {t:6.1f} ms")
    return t

# ── 1. Baseline: JIT cell loop ──
def forward_jit(x):
    bsz = x.shape[0]
    emb = model.embed(x)
    h = torch.zeros(bsz, dm, device=device)
    logits = torch.zeros(bsz, seq, vocab_size, device=device)
    for t in range(seq - 1):
        if t % ae == (ae - 1):
            h = _cell_full(h, emb[:, t], pat, beta_t, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn)
        else:
            h = _cell_ssm(h, emb[:, t], wa, ba, wb, bb, wp, bp, wn, bn)
        logits[:, t] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    h = _cell_full(h + slot[x[:, -1]], emb[:, -1], pat, beta_t, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn)
    logits[:, -1] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    return logits

# ── 2. torch.compile on eager cell loop ──
@torch.compile(dynamic=False, backend="aot_eager")
def forward_compiled(x):
    bsz = x.shape[0]
    emb = model.embed(x)
    h = torch.zeros(bsz, dm, device=device)
    logits = torch.zeros(bsz, seq, vocab_size, device=device)
    for t in range(seq - 1):
        if t % ae == (ae - 1):
            h = _cell_full(h, emb[:, t], pat, beta_t, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn)
        else:
            h = _cell_ssm(h, emb[:, t], wa, ba, wb, bb, wp, bp, wn, bn)
        logits[:, t] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    h = _cell_full(h + slot[x[:, -1]], emb[:, -1], pat, beta_t, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn)
    logits[:, -1] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    return logits

# ── 3. Compile individual cells (plain PyTorch, no JIT) ──
def cell_full_eager(h, x):
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ wa.t() + ba)
    b = torch.sigmoid(combined @ wb.t() + bb)
    h_ssm = a * h + b * (x @ wp.t() + bp)
    scores = (h_ssm @ pat.t()) * beta_t[0]
    attn = torch.softmax(scores, dim=-1)
    alpha_val = torch.sigmoid(combined @ wg.t() + bg)
    h_new = h_ssm + alpha_val * (attn @ pat - h_ssm)
    return F.layer_norm(h_new, [dm], wn, bn, 1e-5)

def cell_ssm_eager(h, x):
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ wa.t() + ba)
    b = torch.sigmoid(combined @ wb.t() + bb)
    h_ssm = a * h + b * (x @ wp.t() + bp)
    return F.layer_norm(h_ssm, [dm], wn, bn, 1e-5)

@torch.compile(dynamic=False, backend="aot_eager")
def forward_compiled_cells(x):
    bsz = x.shape[0]
    emb = model.embed(x)
    h = torch.zeros(bsz, dm, device=device)
    logits = torch.zeros(bsz, seq, vocab_size, device=device)
    for t in range(seq - 1):
        if t % ae == (ae - 1):
            h = cell_full_eager(h, emb[:, t])
        else:
            h = cell_ssm_eager(h, emb[:, t])
        logits[:, t] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    h = cell_full_eager(h + slot[x[:, -1]], emb[:, -1])
    logits[:, -1] = F.layer_norm(h, [dm], snw, snb, 1e-5) @ hw.t() + hb
    return logits

print("Benchmark (vocab=1024, dm=256, np=4096, seq=64, bs=8):")
print()
t_jit  = bench("JIT cell loop",         forward_jit,            x, tgt)
t_cmp  = bench("torch.compile (JIT cells)",  forward_compiled,       x, tgt)
t_cell = bench("torch.compile (eager cells)", forward_compiled_cells, x, tgt)
print()
print(f"Speedup (compiled jit cells):  {t_jit/t_cmp:.2f}x")
print(f"Speedup (compiled eager cells): {t_jit/t_cell:.2f}x")
