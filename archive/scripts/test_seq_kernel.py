"""Test sequence kernel."""
import ctypes, os, torch
device = "cuda"

dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "modules", "cann_step.dll"))
dll.launch_cann_sequence.restype = None
dll.launch_cann_sequence.argtypes = (
    [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_float]
)

dm, np_, bs, seq, vs = 64, 256, 2, 8, 23
h_init = torch.zeros(bs, dm, device=device)
emb = torch.randn(bs, seq, dm, device=device)
p = torch.randn(np_, dm, device=device)
st = torch.zeros(vs, dm, device=device)


def mk(od, id_):
    return torch.randn(od, id_, device=device), torch.randn(od, device=device)


wa, ba = mk(dm, dm * 2)
wb, bb = mk(dm, dm * 2)
wg, bg = mk(dm, dm * 2)
wp, bp = mk(dm, dm)
wn = torch.randn(dm, device=device)
bn = torch.randn(dm, device=device)
hw, hb = mk(vs, dm)

logits = torch.zeros(bs, seq, vs, device=device)
dll.launch_cann_sequence(
    ctypes.c_void_p(h_init.data_ptr()),
    ctypes.c_void_p(emb.data_ptr()),
    ctypes.c_void_p(p.data_ptr()),
    ctypes.c_void_p(st.data_ptr()),
    ctypes.c_void_p(wa.data_ptr()),
    ctypes.c_void_p(ba.data_ptr()),
    ctypes.c_void_p(wb.data_ptr()),
    ctypes.c_void_p(bb.data_ptr()),
    ctypes.c_void_p(wg.data_ptr()),
    ctypes.c_void_p(bg.data_ptr()),
    ctypes.c_void_p(wp.data_ptr()),
    ctypes.c_void_p(bp.data_ptr()),
    ctypes.c_void_p(wn.data_ptr()),
    ctypes.c_void_p(bn.data_ptr()),
    ctypes.c_void_p(hw.data_ptr()),
    ctypes.c_void_p(hb.data_ptr()),
    ctypes.c_void_p(logits.data_ptr()),
    bs,
    seq,
    dm,
    np_,
    vs,
    ctypes.c_float(0.5),
)
print(f"OK shape={logits.shape} range=[{logits.min().item():.2f},{logits.max().item():.2f}]")
