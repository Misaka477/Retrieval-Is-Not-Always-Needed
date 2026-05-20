"""Profile CANN-SSM and test JIT compilation."""
import torch, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel

device = "cuda"
m = RINASeqModel(23, d_model=64, n_patterns=1024, beta=0.5).to(device)

x = torch.randint(0, 23, (16, 35), device=device)

# Python baseline
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100): m(x)
torch.cuda.synchronize()
t_py = (time.time() - t0) / 100 * 1000
print(f"Python forward: {t_py:.1f}ms")

# JIT the cell
@torch.jit.script
def cell_forward_s(
    h: torch.Tensor,
    x: torch.Tensor,
    patterns: torch.Tensor,
    beta: torch.Tensor,
    w_a: torch.Tensor,
    b_a: torch.Tensor,
    w_b: torch.Tensor,
    b_b: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
    w_p: torch.Tensor,
    b_p: torch.Tensor,
    w_n: torch.Tensor,
    b_n: torch.Tensor,
):
    bsz = h.shape[0]
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ w_a.t() + b_a)
    b = torch.sigmoid(combined @ w_b.t() + b_b)
    x_proj = x @ w_p.t() + b_p
    h_ssm = a * h + b * x_proj

    pat = patterns.unsqueeze(0).expand(bsz, -1, -1)
    xi = h_ssm.unsqueeze(1)
    scores = xi @ pat.transpose(1, 2) * beta[0]
    attn = torch.softmax(scores, dim=-1)
    h_attracted = (attn @ pat).squeeze(1)

    alpha = torch.sigmoid(combined @ w_g.t() + b_g)
    h_new = h_ssm + alpha * (h_attracted - h_ssm)
    return torch.layer_norm(h_new, [h.shape[-1]], w_n, b_n, eps=1e-5)


# Extract weights for JIT
cell = m.cell
w_a, b_a = cell.gate_a.weight, cell.gate_a.bias
w_b, b_b = cell.gate_b.weight, cell.gate_b.bias
w_g, b_g = cell.gate_alpha.weight, cell.gate_alpha.bias
w_p, b_p = cell.proj_in.weight, cell.proj_in.bias
w_n, b_n = cell.norm.weight, cell.norm.bias

h = torch.zeros(16, 64, device=device)
emb = m.embed(x)

beta_t = torch.tensor([cell.beta], device=device)

# JIT warmup
for _ in range(10):
    h = torch.zeros(16, 64, device=device)
    for t in range(35):
        h = cell_forward_s(h, emb[:, t], cell.patterns, beta_t,
                           w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)

torch.cuda.synchronize(); t0 = time.time()
for _ in range(100):
    h = torch.zeros(16, 64, device=device)
    for t in range(35):
        h = cell_forward_s(h, emb[:, t], cell.patterns, beta_t,
                           w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)
torch.cuda.synchronize()
t_jit = round((time.time() - t0) / 100 * 1000, 1)
print(f"JIT cell loop: {t_jit:.1f}ms (vs {t_py:.1f}ms Python)")

# Also try JIT on the full sequence
def full_forward(h, emb, patterns, beta_t,
                 w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n):
    bs = emb.shape[0]
    sl = emb.shape[1]
    dm = h.shape[-1]
    out = torch.zeros(bs, sl, dm, device=h.device)
    for t in range(sl):
        h = cell_forward_s(h, emb[:, t], patterns, beta_t,
                           w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)
        out[:, t] = h
    return out

try:
    beta_t = torch.tensor([cell.beta], device=device)
    full_jit = torch.jit.script(full_forward)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100):
        full_jit(torch.zeros(16, 64, device=device), emb, cell.patterns, beta_t,
                 w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)
    torch.cuda.synchronize()
    t_full = (time.time() - t0) / 100 * 1000
    print(f"JIT full loop: {t_full:.1f}ms")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"JIT full loop failed: {e}")
