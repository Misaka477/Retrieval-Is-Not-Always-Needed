"""
Triton kernels for CANN-SSM acceleration.

Phase 1: Infrastructure test + minimal kernel.
Phase 2: Full CANN cell fusion.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def test_kernel(
    x_ptr, y_ptr,
    n_elements: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + 1.0, mask=mask)


def test():
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.zeros(n, device="cuda")

    grid = (triton.cdiv(n, 256),)
    test_kernel[grid](x, y, n, BLOCK=256)

    diff = (y - (x + 1.0)).abs().max().item()
    assert diff < 1e-5, f"Triton test FAIL: max diff {diff:.2e}"
    print(f"Triton test PASS: max diff {diff:.2e}")

    # Speed vs PyTorch
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        test_kernel[grid](x, y, n, BLOCK=256)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 1000 * 1000

    t0 = time.time()
    for _ in range(1000):
        y2 = x + 1.0
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / 1000 * 1000

    print(f"  Triton: {t_triton:.3f}ms  PyTorch: {t_torch:.3f}ms")
    print("Triton infrastructure ready.")


if __name__ == "__main__":
    test()
