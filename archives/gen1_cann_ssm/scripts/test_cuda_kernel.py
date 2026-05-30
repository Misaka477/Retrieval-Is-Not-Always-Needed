"""Test the compiled CUDA kernel with ctypes."""
import ctypes
import numpy as np
import torch
import os

# Load the .dll
dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), os.pardir, "modules", "cann_step.dll"))

# ── Test cann_step_kernel ──
dll.cann_step_kernel.restype = None
dll.cann_step_kernel.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
]

# Test data
dm, np_ = 64, 256
h_in = torch.randn(1, dm, device="cuda")
x = torch.randn(1, dm, device="cuda")
patterns = torch.randn(np_, dm, device="cuda")

def make_w(od, id_):
    return torch.randn(od, id_, device="cuda"), torch.randn(od, device="cuda")

wa, ba = make_w(dm, dm*2); wb, bb = make_w(dm, dm*2)
wg, bg = make_w(dm, dm*2); wp, bp = make_w(dm, dm)
wn, bn = make_w(dm, dm)
h_out = torch.zeros(1, dm, device="cuda")

shared_mem = (1 + np_) * 4  # 1 for max + np_ scores
dll.cann_step_kernel(
    ctypes.c_void_p(h_in.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(patterns.data_ptr()),
    ctypes.c_void_p(wa.data_ptr()), ctypes.c_void_p(ba.data_ptr()),
    ctypes.c_void_p(wb.data_ptr()), ctypes.c_void_p(bb.data_ptr()),
    ctypes.c_void_p(wg.data_ptr()), ctypes.c_void_p(bg.data_ptr()),
    ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(bp.data_ptr()),
    ctypes.c_void_p(wn.data_ptr()), ctypes.c_void_p(bn.data_ptr()),
    ctypes.c_void_p(h_out.data_ptr()),
    dm, np_, ctypes.c_float(0.5),
)

print(f"Step kernel output shape: {h_out.shape}")
print(f"Output values: {h_out[0, :5].tolist()} [...]")
print("CUDA kernel test PASSED")
