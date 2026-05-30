"""Compare h_ssm in CUDA vs PyTorch."""
import ctypes, os, torch

device = "cuda"
dm, np_ = 64, 256
torch.manual_seed(42)

h = torch.randn(1, dm, device=device)
x = torch.randn(1, dm, device=device)
p = torch.randn(np_, dm, device=device)
wa, ba = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wb, bb = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wg, bg = torch.randn(dm, dm*2, device=device), torch.randn(dm, device=device)
wp, bp = torch.randn(dm, dm, device=device), torch.randn(dm, device=device)
wn, bn = torch.randn(dm, device=device), torch.randn(dm, device=device)
beta = 0.5

# PyTorch
combined = torch.cat([h, x], dim=-1)
a = torch.sigmoid(combined @ wa.T + ba)
b = torch.sigmoid(combined @ wb.T + bb)
xp = x @ wp.T + bp
h_ssm_pt = a * h + b * xp
print(f"Pytorch h_ssm head(5): {h_ssm_pt[0,:5].tolist()}")
print(f"Pytorch a head(5):     {a[0,:5].tolist()}")
print(f"Pytorch b head(5):     {b[0,:5].tolist()}")
print(f"Pytorch xp head(5):    {xp[0,:5].tolist()}")

# CUDA
dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "modules", "cann_step.dll")
dll = ctypes.CDLL(dll_path)
dll.launch_cann_step.restype = None
h_out = torch.zeros(1, dm, device=device)
dll.launch_cann_step(
    ctypes.c_void_p(h.data_ptr()), ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(p.data_ptr()),
    ctypes.c_void_p(wa.data_ptr()), ctypes.c_void_p(ba.data_ptr()),
    ctypes.c_void_p(wb.data_ptr()), ctypes.c_void_p(bb.data_ptr()),
    ctypes.c_void_p(wg.data_ptr()), ctypes.c_void_p(bg.data_ptr()),
    ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(bp.data_ptr()),
    ctypes.c_void_p(wn.data_ptr()), ctypes.c_void_p(bn.data_ptr()),
    ctypes.c_void_p(h_out.data_ptr()),
    dm, np_, ctypes.c_float(beta),
)
print(f"CUDA h_out head(5):    {h_out[0,:5].tolist()}")
print(f"Diff h_ssm vs h_out:  {(h_ssm_pt - h_out).abs().max().item():.6f}")
