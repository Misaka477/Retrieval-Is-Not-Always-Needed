"""Benchmark CUDA v2 training speed vs Python loop."""
import torch
import torch.nn.functional as F
import sys, os, time, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, CANNSequenceCUDA, _setup_cuda_seq_v2, _cell_full, _cell_ssm

device = "cuda"
torch.manual_seed(42)

vocab_size = 1024
dm = 256
np_ = 4096
seq = 64
bs = 8
attract_every = 4

model_py = RINASeqModel(vocab_size, d_model=dm, n_patterns=np_,
                         beta=0.5, n_slots=vocab_size, attract_every=attract_every).to(device)
model_cu = copy.deepcopy(model_py)
model_cu.to(device)

x = torch.randint(0, vocab_size, (bs, seq), device=device)
tgt = torch.randint(0, vocab_size, (bs, seq), device=device)

n_warmup = 3
n_iter = 10

# ── Python loop baseline ──
def py_loop(model, x, tgt):
    bsz, seq_len = x.shape
    emb = model.embed(x)
    h = torch.zeros(bsz, dm, device=x.device)
    logits = []
    for t in range(seq_len - 1):
        if t % attract_every == (attract_every - 1):
            h = _cell_full(h, emb[:, t], model.cell.patterns, model.cell.beta_t,
                          model.cell.gate_a.weight, model.cell.gate_a.bias,
                          model.cell.gate_b.weight, model.cell.gate_b.bias,
                          model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                          model.cell.proj_in.weight, model.cell.proj_in.bias,
                          model.cell.norm.weight, model.cell.norm.bias)
        else:
            h = _cell_ssm(h, emb[:, t],
                         model.cell.gate_a.weight, model.cell.gate_a.bias,
                         model.cell.gate_b.weight, model.cell.gate_b.bias,
                         model.cell.proj_in.weight, model.cell.proj_in.bias,
                         model.cell.norm.weight, model.cell.norm.bias)
        logits.append(model.head(model.state_norm(h)))
    i_ext = model.slot_table[x[:, -1]]
    h = _cell_full(h + i_ext, emb[:, -1], model.cell.patterns, model.cell.beta_t,
                   model.cell.gate_a.weight, model.cell.gate_a.bias,
                   model.cell.gate_b.weight, model.cell.gate_b.bias,
                   model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                   model.cell.proj_in.weight, model.cell.proj_in.bias,
                   model.cell.norm.weight, model.cell.norm.bias)
    logits.append(model.head(model.state_norm(h)))
    logits = torch.stack(logits, dim=1)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
    return loss

for _ in range(n_warmup):
    model_py.zero_grad()
    loss = py_loop(model_py, x, tgt)
    loss.backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(n_iter):
    model_py.zero_grad()
    loss = py_loop(model_py, x, tgt)
    loss.backward()
torch.cuda.synchronize()
t_py = (time.time() - t0) / n_iter * 1000
print(f"Python loop: {t_py:.1f} ms/batch (fwd+back)")

# ── CUDA v2 ──
has_cuda = _setup_cuda_seq_v2()
if not has_cuda:
    print("CUDA v2 not available")
    sys.exit(1)

h_init = torch.zeros(bs, dm, device=device)
xt = x.to(torch.int32)
beta = model_cu.cell.beta_t[0].item()
for _ in range(n_warmup):
    model_cu.zero_grad()
    emb = model_cu.embed(x)
    logits = CANNSequenceCUDA.apply(
        h_init, emb, xt,
        model_cu.cell.patterns, model_cu.slot_table,
        model_cu.cell.gate_a.weight, model_cu.cell.gate_a.bias,
        model_cu.cell.gate_b.weight, model_cu.cell.gate_b.bias,
        model_cu.cell.gate_alpha.weight, model_cu.cell.gate_alpha.bias,
        model_cu.cell.proj_in.weight, model_cu.cell.proj_in.bias,
        model_cu.cell.norm.weight, model_cu.cell.norm.bias,
        model_cu.state_norm.weight, model_cu.state_norm.bias,
        model_cu.head.weight, model_cu.head.bias,
        beta, attract_every,
    )
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
    loss.backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(n_iter):
    model_cu.zero_grad()
    emb = model_cu.embed(x)
    logits = CANNSequenceCUDA.apply(
        h_init, emb, xt,
        model_cu.cell.patterns, model_cu.slot_table,
        model_cu.cell.gate_a.weight, model_cu.cell.gate_a.bias,
        model_cu.cell.gate_b.weight, model_cu.cell.gate_b.bias,
        model_cu.cell.gate_alpha.weight, model_cu.cell.gate_alpha.bias,
        model_cu.cell.proj_in.weight, model_cu.cell.proj_in.bias,
        model_cu.cell.norm.weight, model_cu.cell.norm.bias,
        model_cu.state_norm.weight, model_cu.state_norm.bias,
        model_cu.head.weight, model_cu.head.bias,
        beta, attract_every,
    )
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
    loss.backward()
torch.cuda.synchronize()
t_cuda = (time.time() - t0) / n_iter * 1000
print(f"CUDA v2:      {t_cuda:.1f} ms/batch (fwd+back)")
print(f"Speedup:      {t_py / t_cuda:.1f}x")
