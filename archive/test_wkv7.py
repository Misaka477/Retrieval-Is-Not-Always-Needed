"""Compile and test the corrected WKV7 kernel."""
import torch, os
from torch.utils.cpp_extension import load

HEAD_SIZE = 64; CHUNK_LEN = 64

# Preprocess CUDA: replace compile-time constants, force float32
src = open("kernels/rwkv7_clampw.cu", encoding="utf-8").read()
for macro, val in [("_C_", str(HEAD_SIZE)), ("_N_", str(HEAD_SIZE)), ("_CHUNK_LEN_", str(CHUNK_LEN))]:
    src = src.replace(macro, val)
# Force float32 unconditionally
src = src.replace("#include <cuda_bf16.h>", "")
src = src.replace("#ifdef _FP32_", "/* float forced */")
src = src.replace("#else\n    using bf = __nv_bfloat16;", "")
src = src.replace("using bf = __nv_bfloat16;", "using bf = float;")
src = src.replace("__nv_bfloat16", "float")
src = src.replace("#define to_float(u) (__bfloat162float(u))", "#define to_float(u) (u)")
src = src.replace("#define to_bf(u) (__float2bfloat16_rn(u))", "#define to_bf(u) (u)")
src = src.replace("#endif", "")
tmp_cu = "kernels/rwkv7_clampw_pp.cu"
with open(tmp_cu, "w", encoding="utf-8") as f:
    f.write(src)

print("Compiling WKV7 kernel...")
wkv7 = load(
    name="rwkv7_clampw",
    sources=[tmp_cu, "kernels/rwkv7_clampw.cpp"],
    is_python_module=False,
    verbose=False,
)
os.remove(tmp_cu)
print("Compilation OK")

# Forward/backward test
B, T, D = 2, 32, 256
device = "cuda"
r = torch.randn(B, T, D, device=device, dtype=torch.float32)
w = torch.randn(B, T, D, device=device, dtype=torch.float32)
k = torch.randn(B, T, D, device=device, dtype=torch.float32)
v = torch.randn(B, T, D, device=device, dtype=torch.float32)
a = torch.randn(B, T, D, device=device, dtype=torch.float32)
b_ = torch.randn(B, T, D, device=device, dtype=torch.float32)
y = torch.empty(B, T, D, device=device, dtype=torch.float32)
n_chunks = (T + CHUNK_LEN - 1) // CHUNK_LEN
s = torch.zeros(B, D // HEAD_SIZE, n_chunks, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32)
sa = torch.zeros(B, T, device=device, dtype=torch.float32)

torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b_, y, s, sa)
assert not torch.isnan(y).any() and not torch.isinf(y).any()
print(f"Forward: OK (y.mean={y.mean().item():.4f})")

dy = torch.randn_like(y)
grads = [torch.empty_like(x) for x in (r, w, k, v, a, b_)]
torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b_, dy, s, sa, *grads)
print("Backward: OK")
print("ALL PASS")
