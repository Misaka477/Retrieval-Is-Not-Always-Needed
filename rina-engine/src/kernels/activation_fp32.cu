#include <cuda_runtime.h>
#include <cmath>

// SiLU: y[i] = x[i] / (1 + exp(-x[i]))
__global__ void silu_fp32_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void launch_silu_fp32(float* x, int n, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    silu_fp32_kernel<<<grid, block, 0, stream>>>(x, n);
}

// SiLU backward: dout[n], gate[n] → d_gate[n]
// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void silu_bwd_kernel(const float* dout, const float* gate,
    float* d_gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = 1.0f / (1.0f + expf(-g));
    d_gate[i] = dout[i] * s * (1.0f + g * (1.0f - s));
}

void launch_silu_bwd_fp32(const float* dout, const float* gate,
    float* d_gate, int n, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    silu_bwd_kernel<<<grid, block, 0, stream>>>(dout, gate, d_gate, n);
}

// silu_mul: out = silu(gate) * up
// d_gate = d_out * up * silu'(gate)
// d_up = d_out * silu(gate) = d_out * gate * sigmoid(gate)
__global__ void silu_mul_bwd_kernel(const float* dout, const float* gate,
    const float* up, float* d_gate, float* d_up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sg = 1.0f / (1.0f + expf(-g));
    float dsg = sg * (1.0f + g * (1.0f - sg));
    d_gate[i] = dout[i] * up[i] * dsg;
    d_up[i] = dout[i] * g * sg;
}

void launch_silu_mul_bwd_fp32(const float* dout, const float* gate,
    const float* up, float* d_gate, float* d_up,
    int n, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    silu_mul_bwd_kernel<<<grid, block, 0, stream>>>(dout, gate, up, d_gate, d_up, n);
}

// Utility: copy (used by SSM/MLA backward)
void launch_copy_f32(float* dst, const float* src, int n, cudaStream_t stream) {
    if (n > 0) cudaMemcpyAsync(dst, src, n*sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

// Utility: silu_mul in-place (used by SSM/MLA backward)
static const int BLK2 = 256;
__global__ void silu_mul_simple_k(float* o, const float* g, const float* u, int n) {
    int i = blockIdx.x * BLK2 + threadIdx.x;
    if (i < n) o[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}
void launch_silu_mul_inline(float* o, const float* g, const float* u, int n, cudaStream_t stream) {
    silu_mul_simple_k<<<(n+BLK2-1)/BLK2,BLK2,0,stream>>>(o,g,u,n);
}
