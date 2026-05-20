"""Benchmark mixed precision for RINA training."""
import torch
import torch.nn.functional as F
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, _cell_full, _cell_ssm, _full_forward

device = "cuda"
torch.manual_seed(42)

vocab_size = 1024; dm = 256; np_ = 4096; seq = 64; bs = 8; ae = 4
model = RINASeqModel(vocab_size, d_model=dm, n_patterns=np_,
                     beta=0.5, n_slots=vocab_size, attract_every=ae).to(device)

x = torch.randint(0, vocab_size, (bs, seq), device=device)
tgt = torch.randint(0, vocab_size, (bs, seq), device=device)

n_warmup = 5; n_iter = 20

def bench_fp32(name, x, tgt):
    for _ in range(n_warmup):
        model.zero_grad()
        out = _full_forward(x, model.embed.weight, model.slot_table,
                           model.head.weight, model.head.bias,
                           model.state_norm.weight, model.state_norm.bias,
                           model.cell.patterns, model.cell.beta_t,
                           model.cell.gate_a.weight, model.cell.gate_a.bias,
                           model.cell.gate_b.weight, model.cell.gate_b.bias,
                           model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                           model.cell.proj_in.weight, model.cell.proj_in.bias,
                           model.cell.norm.weight, model.cell.norm.bias, ae)
        loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model.zero_grad()
        out = _full_forward(x, model.embed.weight, model.slot_table,
                           model.head.weight, model.head.bias,
                           model.state_norm.weight, model.state_norm.bias,
                           model.cell.patterns, model.cell.beta_t,
                           model.cell.gate_a.weight, model.cell.gate_a.bias,
                           model.cell.gate_b.weight, model.cell.gate_b.bias,
                           model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                           model.cell.proj_in.weight, model.cell.proj_in.bias,
                           model.cell.norm.weight, model.cell.norm.bias, ae)
        loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t = (time.time() - t0) / n_iter * 1000
    print(f"  {name:35s} {t:6.1f} ms")
    return t

def bench_fp16(name, x, tgt):
    for _ in range(n_warmup):
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            out = _full_forward(x, model.embed.weight, model.slot_table,
                               model.head.weight, model.head.bias,
                               model.state_norm.weight, model.state_norm.bias,
                               model.cell.patterns, model.cell.beta_t,
                               model.cell.gate_a.weight, model.cell.gate_a.bias,
                               model.cell.gate_b.weight, model.cell.gate_b.bias,
                               model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                               model.cell.proj_in.weight, model.cell.proj_in.bias,
                               model.cell.norm.weight, model.cell.norm.bias, ae)
            loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            out = _full_forward(x, model.embed.weight, model.slot_table,
                               model.head.weight, model.head.bias,
                               model.state_norm.weight, model.state_norm.bias,
                               model.cell.patterns, model.cell.beta_t,
                               model.cell.gate_a.weight, model.cell.gate_a.bias,
                               model.cell.gate_b.weight, model.cell.gate_b.bias,
                               model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                               model.cell.proj_in.weight, model.cell.proj_in.bias,
                               model.cell.norm.weight, model.cell.norm.bias, ae)
            loss = F.cross_entropy(out.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
    torch.cuda.synchronize()
    t = (time.time() - t0) / n_iter * 1000
    print(f"  {name:35s} {t:6.1f} ms")
    return t

print("Mixed precision (vocab=1024, dm=256, np=4096, seq=64, bs=8):")
print()
t32 = bench_fp32("FP32 JIT (fwd+back)", x, tgt)
t16 = bench_fp16("FP16 AMP (fwd+back)", x, tgt)
print()
print(f"FP16 speedup: {t32/t16:.2f}x")
