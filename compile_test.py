import os, torch
# Add MSVC to PATH so ninja/cl can find it
os.environ["PATH"] = r"D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + os.environ["PATH"]
os.environ["PATH"] = r"D:\Software_Development\CUDA_Toolkit_12.4\bin;" + os.environ["PATH"]

from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void fill_kernel(float* x, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}
torch::Tensor fill(torch::Tensor x, float val) {
    int n = x.numel();
    fill_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), val, n);
    return x;
}
"""

mod = load_inline(name="fill_test", cpp_sources="torch::Tensor fill(torch::Tensor x, float val);",
                  cuda_sources=cuda_src, functions=["fill"], verbose=False)
x = torch.zeros(100, device="cuda")
mod.fill(x, 42.0)
print("CUDA OK:", x[:5].tolist())
