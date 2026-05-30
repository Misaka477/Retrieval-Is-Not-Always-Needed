"""Verify CUDA kernel output matches PyTorch CANN cell."""
import ctypes, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "modules", "cann_step.dll")
dll = ctypes.CDLL(dll_path)
dll.launch_cann_step.restype = None
dll.launch_cann_step.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]

device = "cuda"
dm, np_ = 64, 256
torch.manual_seed(42)

h = torch.randn(1, dm, device=device)
x = torch.randn(1, dm, device=device)
patterns = torch.randn(np_, dm, device=device)
wa, ba = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wb, bb = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wg, bg = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wp, bp = torch.randn(dm, dm, device=device), torch.randn(dm, device=device)
wn, bn = torch.randn(dm, device=device), torch.randn(dm, device=device)
beta = 0.5

# ── PyTorch reference ──
def cann_cell_pytorch(h, x):
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ wa.T + ba)
    b = torch.sigmoid(combined @ wb.T + bb)
    xp = x @ wp.T + bp
    h_ssm = a * h + b * xp

    scores = (h_ssm @ patterns.T) * beta
    attn = torch.softmax(scores, dim=-1)
    attracted = attn @ patterns

    alpha = torch.sigmoid(combined @ wg.T + bg)
    h_new = h_ssm + alpha * (attracted - h_ssm)
    mean = h_new.mean(dim=-1, keepdim=True)
    var = h_new.var(dim=-1, keepdim=True, unbiased=False)
    h_norm = wn * (h_new - mean) / torch.sqrt(var + 1e-5) + bn
    return h_norm

h_ref = cann_cell_pytorch(h, x)

# ── CUDA kernel ──
h_out = torch.zeros(1, dm, device=device)
dll.launch_cann_step(
    ctypes.c_void_p(h.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(patterns.data_ptr()),
    ctypes.c_void_p(wa.data_ptr()), ctypes.c_void_p(ba.data_ptr()),
    ctypes.c_void_p(wb.data_ptr()), ctypes.c_void_p(bb.data_ptr()),
    ctypes.c_void_p(wg.data_ptr()), ctypes.c_void_p(bg.data_ptr()),
    ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(bp.data_ptr()),
    ctypes.c_void_p(wn.data_ptr()), ctypes.c_void_p(bn.data_ptr()),
    ctypes.c_void_p(h_out.data_ptr()),
    dm, np_, ctypes.c_float(beta),
)

diff = (h_out - h_ref).abs().max().item()
print(f"Max diff: {diff:.6f}")
print(f"CUDA  head: {h_out[0, :5].tolist()}")
print(f"Torch head: {h_ref[0, :5].tolist()}")
if diff < 1e-3:
    print("PASS: CUDA kernel matches PyTorch")
else:
    print(f"FAIL: diff too large ({diff})")
