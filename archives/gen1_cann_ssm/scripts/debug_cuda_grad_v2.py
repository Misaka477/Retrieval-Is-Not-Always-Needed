"""Verify CUDA sequence v2 gradients vs Python loop reference."""
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, CANNSequenceCUDA, _setup_cuda_seq_v2

device = "cuda"
torch.manual_seed(42)

vocab_size = 32
dm = 64
np_ = 128
seq = 16
bs = 2
attract_every = 4

model = RINASeqModel(vocab_size, d_model=dm, n_patterns=np_,
                     beta=0.5, n_slots=32, attract_every=attract_every).to(device)
model.train()

x = torch.randint(0, vocab_size, (bs, seq), device=device)
tgt = torch.randint(0, vocab_size, (bs, seq), device=device)

# Clone all parameters
params_py = {name: p.clone().detach().requires_grad_(True)
             for name, p in model.named_parameters()}
# Assign same values to original model
for name, p in model.named_parameters():
    p.data.copy_(params_py[name].data)

# ── Python loop reference ──
emb_ref = model.embed(x)
h_ref = torch.zeros(bs, dm, device=device)
logits_ref = []
for t in range(seq - 1):
    h_ref = model.cell(h_ref, emb_ref[:, t, :], step=t)
    logits_ref.append(model.head(model.state_norm(h_ref)))
i_ext = model.slot_table[x[:, -1]]
h_ref = model.cell(h_ref + i_ext, emb_ref[:, -1, :], step=seq - 1)
logits_ref.append(model.head(model.state_norm(h_ref)))
logits_ref = torch.stack(logits_ref, dim=1)

loss_ref = F.cross_entropy(logits_ref.reshape(-1, vocab_size), tgt.reshape(-1))
loss_ref.backward()

ref_grads = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        ref_grads[name] = p.grad.clone()
    else:
        ref_grads[name] = torch.zeros_like(p)
model.zero_grad()

# ── CUDA sequence v2 ──
has_cuda = _setup_cuda_seq_v2()
if not has_cuda:
    print("ERROR: CUDA DLL not loaded")
    sys.exit(1)

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

loss_cuda = F.cross_entropy(logits_cuda.reshape(-1, vocab_size), tgt.reshape(-1))
loss_cuda.backward()

cuda_grads = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        cuda_grads[name] = p.grad.clone()
    else:
        cuda_grads[name] = torch.zeros_like(p)

# ── Compare ──
print(f"Logits max diff: {(logits_ref - logits_cuda).abs().max().item():.6f}")
print(f"Loss: ref={loss_ref.item():.6f}, cuda={loss_cuda.item():.6f}")
print()
print("Gradient comparison:")
all_ok = True
for name in sorted(ref_grads.keys()):
    rg = ref_grads[name]
    cg = cuda_grads[name]
    diff = (rg - cg).abs()
    max_d = diff.max().item()
    rel_d = (diff.mean() / (rg.abs().mean() + 1e-8)).item()
    status = "OK" if max_d < 1e-2 else "MISMATCH"
    if max_d >= 1e-2:
        all_ok = False
    print(f"  {name:30s} max_diff={max_d:.6f}  mean_rel={rel_d:.6f}  [{status}]")

print()
if all_ok:
    print("ALL GRADIENTS MATCH (max diff < 1e-2)")
else:
    print("SOME GRADIENTS MISMATCH - investigation needed")
