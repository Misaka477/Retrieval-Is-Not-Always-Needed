"""Debug: compare d_h_ssm, d_emb, etc. for a tiny sequence."""
import torch
import torch.nn.functional as F
import sys, os, ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, CANNSequenceCUDA, _setup_cuda_seq_v2
from modules.cann_ssm import _cell_full, _cell_ssm

device = "cuda"
torch.manual_seed(1)

vocab_size = 8
dm = 16
np_ = 32
seq = 2
bs = 1
attract_every = 1  # always full attractor

model = RINASeqModel(vocab_size, d_model=dm, n_patterns=np_,
                     beta=0.5, n_slots=vocab_size, attract_every=attract_every).to(device)
model.train()

x = torch.tensor([[2, 5]], device=device)
tgt = torch.tensor([[3, 7]], device=device)

# Zero out slot_table for simplicity
model.slot_table.zero_()

# ── Python reference ──
emb_ref = model.embed(x)
h = torch.zeros(bs, dm, device=device)

# Step 0
h_after_0 = _cell_full(
    h, emb_ref[:, 0],
    model.cell.patterns, model.cell.beta_t,
    model.cell.gate_a.weight, model.cell.gate_a.bias,
    model.cell.gate_b.weight, model.cell.gate_b.bias,
    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
    model.cell.proj_in.weight, model.cell.proj_in.bias,
    model.cell.norm.weight, model.cell.norm.bias,
)
log0 = model.head(model.state_norm(h_after_0))

# Step 1 (last)
h_after_1 = _cell_full(
    h_after_0, emb_ref[:, 1],
    model.cell.patterns, model.cell.beta_t,
    model.cell.gate_a.weight, model.cell.gate_a.bias,
    model.cell.gate_b.weight, model.cell.gate_b.bias,
    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
    model.cell.proj_in.weight, model.cell.proj_in.bias,
    model.cell.norm.weight, model.cell.norm.bias,
)
log1 = model.head(model.state_norm(h_after_1))

logits_ref = torch.stack([log0, log1], dim=1)  # [1, 2, vocab_size]

loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), tgt.view(-1))
loss_ref.backward()
ref_grads = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
             for n, p in model.named_parameters()}
model.zero_grad()

# ── CUDA v2 ──
_setup_cuda_seq_v2()
h_init = torch.zeros(bs, dm, device=device)
emb = model.embed(x)
xt = x.to(torch.int32)
beta = model.cell.beta_t[0].item()

logits_cuda = CANNSequenceCUDA.apply(
    h_init, emb, xt,
    model.cell.patterns, model.slot_table,
    model.cell.gate_a.weight, model.cell.gate_a.bias,
    model.cell.gate_b.weight, model.cell.gate_b.bias,
    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
    model.cell.proj_in.weight, model.cell.proj_in.bias,
    model.cell.norm.weight, model.cell.norm.bias,
    model.state_norm.weight, model.state_norm.bias,
    model.head.weight, model.head.bias,
    beta, model.attract_every,
)

loss_cuda = F.cross_entropy(logits_cuda.view(-1, vocab_size), tgt.view(-1))
loss_cuda.backward()
cuda_grads = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
              for n, p in model.named_parameters()}

print(f"Logits forward diff: {(logits_ref - logits_cuda).abs().max().item():.6f}")
print(f"Loss ref={loss_ref.item():.6f} cuda={loss_cuda.item():.6f}")
print()

# Compare each param gradient
for name in sorted(ref_grads.keys()):
    rg = ref_grads[name]
    cg = cuda_grads[name]
    diff = (rg - cg).abs()
    max_d = diff.max().item()
    mean_r = rg.abs().mean().item()
    rel_d = diff.mean().item() / (mean_r + 1e-8)
    print(f"  {name:30s} ref_mean={mean_r:.6f}  max_diff={max_d:.6f}  rel_mean={rel_d:.6f}")
