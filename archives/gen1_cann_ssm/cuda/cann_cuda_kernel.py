"""
CUDA C kernel for CANN-SSM forward pass.

Compiles with nvcc via torch.utils.cpp_extension.load_inline.
Fuses the per-token loop into a single kernel.
"""
import torch
from torch.utils.cpp_extension import load_inline


_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ── CANN-SSM forward kernel ─────────────────────────────────────
// Processes the full sequence in one kernel launch.
//
// Grid: (batch_size, seq_len, d_model) 
// Each thread handles one element: (b, t, d)

__global__ void cann_forward_kernel(
    const float* __restrict__ h_init,   // (batch, d_model) — initial state
    const float* __restrict__ emb,      // (batch, seq_len, d_model) — token embs
    const float* __restrict__ patterns, // (n_patterns, d_model)
    const float* __restrict__ slot_table, // (vocab, d_model)
    const float* __restrict__ w_a, const float* __restrict__ b_a,
    const float* __restrict__ w_b, const float* __restrict__ b_b,
    const float* __restrict__ w_g, const float* __restrict__ b_g,
    const float* __restrict__ w_p, const float* __restrict__ b_p,
    const float* __restrict__ w_n, const float* __restrict__ b_n,
    const float* __restrict__ head_w, const float* __restrict__ head_b,
    float* __restrict__ logits_out,    // (batch, seq_len, vocab)
    int d_model, int n_patterns, int seq_len, int vocab_size,
    float beta
) {
    int b = blockIdx.x;  // batch
    int d = threadIdx.x; // feature dim

    // Each block handles one batch item
    extern __shared__ float shared[];
    float* h_state = shared;          // (d_model)
    float* h_ssm = shared + d_model;  // (d_model)
    float* scores = shared + 2 * d_model; // (n_patterns)

    // Initialize state from h_init
    if (d < d_model) h_state[d] = h_init[b * d_model + d];
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float* x = &emb[b * seq_len * d_model + t * d_model];

        // ── Step 1: SSM gates ──
        // gate_a: sigmoid(W_a * cat(h, x))
        // gate_b: sigmoid(W_b * cat(h, x))
        float cat[256]; // max d_model
        float gate_a_val = 0, gate_b_val = 0;

        for (int k = 0; k < d_model; k++) {
            cat[k] = h_state[k];
            cat[d_model + k] = x[k];
        }
        __syncthreads();

        // Compute gate for this thread's dimension
        float a_sum = 0, b_sum = 0;
        for (int k = 0; k < d_model * 2; k++) {
            a_sum += w_a[d * (d_model * 2) + k] * cat[k];
            b_sum += w_b[d * (d_model * 2) + k] * cat[k];
        }
        a_sum += b_a[d];
        b_sum += b_b[d];

        float a = 1.0f / (1.0f + expf(-a_sum)); // sigmoid
        float b_gate = 1.0f / (1.0f + expf(-b_sum));

        // h_ssm = a * h + b * proj_in(x)
        float x_proj = 0;
        for (int k = 0; k < d_model; k++) {
            x_proj += w_p[d * d_model + k] * x[k];
        }
        x_proj += b_p[d];
        h_ssm[d] = a * h_state[d] + b_gate * x_proj;

        // ── Step 2: Attractor ──
        // scores[d] = pattern[d] @ h_ssm
        if (d < n_patterns) {
            float s = 0;
            for (int k = 0; k < d_model; k++) {
                s += patterns[d * d_model + k] * h_ssm[k];
            }
            scores[d] = s * beta;
        }
        __syncthreads();

        // softmax over n_patterns (reduce within block)
        if (d < n_patterns) {
            float max_val = -FLT_MAX;
            for (int i = 0; i < n_patterns; i++) {
                max_val = fmaxf(max_val, scores[i]);
            }
            float sum = 0;
            for (int i = 0; i < n_patterns; i++) {
                scores[i] = expf(scores[i] - max_val);
                sum += scores[i];
            }
            for (int i = 0; i < n_patterns; i++) {
                scores[i] /= sum;
            }
        }
        __syncthreads();

        // attracted = patterns^T @ softmax(scores)
        float attracted_val = 0;
        for (int p = 0; p < n_patterns; p++) {
            attracted_val += patterns[p * d_model + d] * scores[p];
        }

        // alpha gate
        float alpha_sum = 0;
        for (int k = 0; k < d_model * 2; k++) {
            alpha_sum += w_g[d * (d_model * 2) + k] * cat[k];
        }
        alpha_sum += b_g[d];
        float alpha = 1.0f / (1.0f + expf(-alpha_sum));

        // h_new = h_ssm + alpha * (attracted - h_ssm)
        float h_new = h_ssm[d] + alpha * (attracted_val - h_ssm[d]);

        // ── Step 3: LayerNorm ──
        float mean = 0, var = 0;
        for (int k = 0; k < d_model; k++) mean += h_ssm[k];  // use old h_ssm for norm
        mean /= d_model;
        for (int k = 0; k < d_model; k++) {
            float diff = h_ssm[k] - mean;
            var += diff * diff;
        }
        var /= d_model;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        float h_norm = w_n[d] * (h_new - mean) * inv_std + b_n[d];

        // ── Step 4: Head projection ──
        // Logits only at the last position + slot injection
        if (t == seq_len - 1) {
            // Slot injection
            int tid = (int)x[0]; // first element = token id (simplified)
            float slot_val[256]; // max d_model
            for (int k = 0; k < d_model; k++) {
                slot_val[k] = slot_table[tid * d_model + k];
            }
            float h_injected = h_norm + slot_val[d]; // d-th dim of slot

            // Head
            float logit = 0;
            for (int k = 0; k < d_model; k++) {
                logit += head_w[d * d_model + k] * h_injected;
            }
            logit += head_b[d];
            logits_out[b * seq_len * vocab_size + t * vocab_size + d] = logit;
        }

        // Update h_state for next step
        h_state[d] = h_new;
        __syncthreads();
    }
}


// ── Host wrapper ──
torch::Tensor cann_forward(
    torch::Tensor h_init,
    torch::Tensor emb,
    torch::Tensor patterns,
    torch::Tensor slot_table,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor w_b, torch::Tensor b_b,
    torch::Tensor w_g, torch::Tensor b_g,
    torch::Tensor w_p, torch::Tensor b_p,
    torch::Tensor w_n, torch::Tensor b_n,
    torch::Tensor head_w, torch::Tensor head_b,
    float beta
) {
    int batch = emb.size(0);
    int seq_len = emb.size(1);
    int d_model = emb.size(2);
    int n_patterns = patterns.size(0);
    int vocab_size = head_w.size(0);

    auto logits = torch::zeros({batch, seq_len, vocab_size}, emb.options());

    int shared_mem = (2 * d_model + n_patterns) * sizeof(float);

    dim3 grid(batch);
    dim3 block(d_model);

    cann_forward_kernel<<<grid, block, shared_mem>>>(
        h_init.data_ptr<float>(),
        emb.data_ptr<float>(),
        patterns.data_ptr<float>(),
        slot_table.data_ptr<float>(),
        w_a.data_ptr<float>(), b_a.data_ptr<float>(),
        w_b.data_ptr<float>(), b_b.data_ptr<float>(),
        w_g.data_ptr<float>(), b_g.data_ptr<float>(),
        w_p.data_ptr<float>(), b_p.data_ptr<float>(),
        w_n.data_ptr<float>(), b_n.data_ptr<float>(),
        head_w.data_ptr<float>(), head_b.data_ptr<float>(),
        logits.data_ptr<float>(),
        d_model, n_patterns, seq_len, vocab_size, beta
    );

    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cann_forward, "CANN-SSM forward kernel");
}
"""

_cpp_source = """
torch::Tensor cann_forward(
    torch::Tensor h_init, torch::Tensor emb, torch::Tensor patterns,
    torch::Tensor slot_table,
    torch::Tensor w_a, torch::Tensor b_a,
    torch::Tensor w_b, torch::Tensor b_b,
    torch::Tensor w_g, torch::Tensor b_g,
    torch::Tensor w_p, torch::Tensor b_p,
    torch::Tensor w_n, torch::Tensor b_n,
    torch::Tensor head_w, torch::Tensor head_b,
    float beta
);
"""


def load_cann_kernel():
    """Compile and load the CUDA kernel."""
    try:
        cann_kernel = load_inline(
            name="cann_cuda",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["forward"],
            verbose=False,
        )
        return cann_kernel
    except Exception as e:
        print(f"CUDA kernel compilation failed: {e}")
        print("Falling back to PyTorch implementation.")
        return None
