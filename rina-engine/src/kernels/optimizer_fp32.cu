#include <cuda_runtime.h>
#include <cmath>

// AdamW: in-place weight update
// m = beta1*m + (1-beta1)*grad
// v = beta2*v + (1-beta2)*grad^2
// m_hat = m / (1-beta1^step)
// v_hat = v / (1-beta2^step)
// w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
__global__ void adamw_kernel(float* w, const float* grad,
    float* m, float* v, float lr, float beta1, float beta2,
    float beta1_pow, float beta2_pow, float eps, float wd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float m_i = beta1 * m[i] + (1.0f - beta1) * g;
    float v_i = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_i;
    v[i] = v_i;
    float m_hat = m_i / (1.0f - beta1_pow);
    float v_hat = v_i / (1.0f - beta2_pow);
    w[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * w[i]);
}

void launch_adamw_fp32(float* w, const float* grad,
    float* m, float* v, int n, float lr,
    float beta1, float beta2,
    float beta1_pow, float beta2_pow,
    float eps, float wd, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    adamw_kernel<<<grid, block, 0, stream>>>(w, grad, m, v, lr,
        beta1, beta2, beta1_pow, beta2_pow, eps, wd, n);
}
