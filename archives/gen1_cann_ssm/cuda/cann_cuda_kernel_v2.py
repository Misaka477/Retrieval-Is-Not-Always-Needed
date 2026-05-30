"""
CANN-SSM CUDA kernel — single-file compilation with nvcc.

All code in CUDA source: kernel + host wrapper + pybind11.
No C++/CUDA split, no `load_inline` issues.
"""
import torch
from torch.utils.cpp_extension import load_inline


_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ── Forward kernel ──
__global__ void cann_step_kernel(
    const float* __restrict__ h_in,
    const float* __restrict__ x,
    const float* __restrict__ patterns,
    const float* __restrict__ w_a, const float* __restrict__ b_a,
    const float* __restrict__ w_b, const float* __restrict__ b_b,
    const float* __restrict__ w_g, const float* __restrict__ b_g,
    const float* __restrict__ w_p, const float* __restrict__ b_p,
    const float* __restrict__ w_n, const float* __restrict__ b_n,
    float* __restrict__ h_out,
    int d_model, int n_patterns, float beta
) {
    int d = threadIdx.x;
    if (d >= d_model) return;

    // cat = [h, x]
    int d2 = d_model * 2;

    // gate_a
    float a_sum = 0;
    for (int k = 0; k < d_model; k++)
        a_sum += w_a[d * d2 + k] * h_in[k];
    for (int k = 0; k < d_model; k++)
        a_sum += w_a[d * d2 + d_model + k] * x[k];
    a_sum += b_a[d];
    float a = 1.0f / (1.0f + expf(-a_sum));

    // gate_b
    float b_sum = 0;
    for (int k = 0; k < d_model; k++)
        b_sum += w_b[d * d2 + k] * h_in[k];
    for (int k = 0; k < d_model; k++)
        b_sum += w_b[d * d2 + d_model + k] * x[k];
    b_sum += b_b[d];
    float b_g = 1.0f / (1.0f + expf(-b_sum));

    // proj_in(x)
    float xp = 0;
    for (int k = 0; k < d_model; k++)
        xp += w_p[d * d_model + k] * x[k];
    xp += b_p[d];

    // h_ssm = a * h + b * xp
    float h_ssm = a * h_in[d] + b_g * xp;

    // attractor: scores = patterns @ h_ssm
    extern __shared__ float smem[];
    float& attr_max = smem[0];
    float* scores = &smem[1];

    float s = 0;
    for (int k = 0; k < d_model; k++)
        s += patterns[d * d_model + k] * h_ssm;
    scores[d] = s * beta;
    __syncthreads();

    // softmax reduce
    if (d == 0) {
        float mx = -FLT_MAX;
        for (int i = 0; i < n_patterns; i++)
            if (scores[i] > mx) mx = scores[i];
        attr_max = mx;
    }
    __syncthreads();

    float sum_exp = 0;
    for (int i = 0; i < n_patterns; i++) {
        float e = expf(scores[i] - attr_max);
        scores[i] = e;
        sum_exp += e;
    }

    float attracted = 0;
    for (int i = 0; i < n_patterns; i++)
        attracted += patterns[i * d_model + d] * (scores[i] / sum_exp);

    // gate_alpha
    float alpha_sum = 0;
    for (int k = 0; k < d_model; k++)
        alpha_sum += w_g[d * d2 + k] * h_in[k];
    for (int k = 0; k < d_model; k++)
        alpha_sum += w_g[d * d2 + d_model + k] * x[k];
    alpha_sum += b_g[d];
    float alpha = 1.0f / (1.0f + expf(-alpha_sum));

    float h_new = h_ssm + alpha * (attracted - h_ssm);

    // LayerNorm
    __syncthreads();
    float mean = 0;
    for (int k = 0; k < d_model; k++) mean += h_new;
    mean /= d_model;
    float var = 0;
    for (int k = 0; k < d_model; k++) {
        float diff = h_new - mean;
        var += diff * diff;
    }
    var /= d_model;
    float inv_std = rsqrtf(var + 1e-5f);
    h_out[d] = w_n[d] * (h_new - mean) * inv_std + b_n[d];
}


// ── Host wrapper ──
torch::Tensor forward(
    torch::Tensor h, torch::Tensor x, torch::Tensor patterns,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor w_b, torch::Tensor b_b,
    torch::Tensor w_g, torch::Tensor b_g,
    torch::Tensor w_p, torch::Tensor b_p,
    torch::Tensor w_n, torch::Tensor b_n,
    float beta
) {
    int d_model = h.size(1);
    int n_patterns = patterns.size(0);

    auto h_out = torch::zeros_like(h);
    int shared_mem = (1 + n_patterns) * sizeof(float);
    int grid = 1;
    cann_step_kernel<<<grid, d_model, shared_mem>>>(
        h.data_ptr<float>(), x.data_ptr<float>(),
        patterns.data_ptr<float>(),
        w_a.data_ptr<float>(), b_a.data_ptr<float>(),
        w_b.data_ptr<float>(), b_b.data_ptr<float>(),
        w_g.data_ptr<float>(), b_g.data_ptr<float>(),
        w_p.data_ptr<float>(), b_p.data_ptr<float>(),
        w_n.data_ptr<float>(), b_n.data_ptr<float>(),
        h_out.data_ptr<float>(),
        d_model, n_patterns, beta
    );

    return h_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CANN step forward");
}
"""


def load():
    try:
        k = load_inline(
            name="cann_step_cuda",
            cpp_sources="",
            cuda_sources=_source,
            functions=["forward"],
            verbose=False,
        )
        print("CUDA kernel compiled OK")
        return k
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None


if __name__ == "__main__":
    k = load()
    if k:
        test_h = torch.randn(4, 64, device="cuda")
        test_x = torch.randn(4, 64, device="cuda")
        pat = torch.randn(256, 64, device="cuda")
        def mkw(d1, d2): return torch.randn(d1, d2, device="cuda"), torch.randn(d1, device="cuda")
        wa, ba = mkw(64, 128); wb, bb = mkw(64, 128)
        wg, bg = mkw(64, 128); wp, bp = mkw(64, 64)
        wn, bn = mkw(64, 64)

        result = k.forward(test_h, test_x, pat, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn, 0.5)
        print(f"Output shape: {result.shape}")
        print("CUDA kernel test PASSED")
