#!/usr/bin/env python3
"""Generate PyTorch reference gradients for each op.
Saves to .npy files that the C++ test loads for comparison."""
import torch, numpy as np, os, sys

OUT = "/tmp/gradcheck_ref"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

def save(name, **kwargs):
    for k, v in kwargs.items():
        arr = np.ascontiguousarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else v.cpu().numpy().astype(np.float32)
        arr.tofile(f"{OUT}/{name}_{k}.bin")
        with open(f"{OUT}/{name}_{k}_shape.txt", 'w') as f:
            f.write(" ".join(str(s) for s in arr.shape))
    print(f"  saved {name}: { {k: (v.shape if hasattr(v,'shape') else 'scalar') for k,v in kwargs.items()} }")

def d(*shape): return torch.randn(*shape, device='cuda', dtype=torch.float64, requires_grad=True)
def f32(t): return t.detach().float().cpu().numpy()
def to_np(t): return t.detach().float().cpu().numpy()

# 1. SiLU
def grad_silu():
    x = d(128, 64)
    out = torch.nn.functional.silu(x)
    loss = out.sum()
    loss.backward()
    save("silu", x=to_np(x), dx=to_np(x.grad))

# 2. SiLU_MUL (out = silu(gate) * up)
def grad_silu_mul():
    gate = d(128, 64)
    up = d(128, 64)
    out = torch.nn.functional.silu(gate) * up
    loss = out.sum()
    loss.backward()
    save("silu_mul", gate=to_np(gate), up=to_np(up),
         dgate=to_np(gate.grad), dup=to_np(up.grad))

# 3. Linear: out = in @ W^T
def grad_linear():
    M, N, K = 32, 64, 128
    x = d(M, K)
    w = d(N, K)
    out = x @ w.T
    loss = out.sum()
    loss.backward()
    save("linear", x=to_np(x), w=to_np(w),
         dx=to_np(x.grad), dw=to_np(w.grad))

# 4. CrossEntropyLoss
def grad_crossentropy():
    N, V = 32, 512
    logits = d(N, V)
    targets = torch.randint(0, V, (N,), device='cuda')
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
    loss.backward()
    save("crossentropy", logits=to_np(logits), 
         targets=targets.cpu().numpy().astype(np.int32),
         dlogits=to_np(logits.grad))

# 5. LayerNorm
def grad_layernorm():
    N, D = 32, 256
    x = d(N, D)
    w = d(D)
    out = torch.nn.functional.layer_norm(x, [D], w, eps=1e-5)
    loss = out.sum()
    loss.backward()
    save("layernorm", x=to_np(x), w=to_np(w),
         dx=to_np(x.grad), dw_=to_np(w.grad))

# 6. Embedding
def grad_embedding():
    B, T, D, V = 1, 32, 256, 512
    weight = d(V, D)
    idx = torch.randint(0, V, (B, T), device='cuda')
    out = torch.nn.functional.embedding(idx, weight)
    loss = out.sum()
    loss.backward()
    save("embedding", weight=to_np(weight), idx=idx.cpu().numpy().astype(np.int32),
         d_weight=to_np(weight.grad))

# 7. RoPE rotation (forward: [c -s; s c] @ [x0; x1])
def grad_rope():
    B, T, H, dim = 2, 16, 4, 32
    half = dim // 2
    x = torch.randn(B*T*H, dim, device='cuda', dtype=torch.float64, requires_grad=True)
    cos = torch.randn(T, half, device='cuda')
    sin = torch.randn(T, half, device='cuda')
    cos.requires_grad = sin.requires_grad = False
    # Vectorized forward
    idx_t = torch.arange(B*T*H, device='cuda') // H % T  # [N] time index per row
    c_t = cos[idx_t]  # [N, half]
    s_t = sin[idx_t]
    x_even = x[:, 0::2]  # [N, half]
    x_odd  = x[:, 1::2]  # [N, half]
    y = torch.zeros_like(x)
    y[:, 0::2] = x_even * c_t - x_odd * s_t
    y[:, 1::2] = x_even * s_t + x_odd * c_t
    loss = y.sum()
    loss.backward()
    save("rope", x=to_np(x), cos=to_np(cos), sin=to_np(sin),
         dx=to_np(x.grad))

# 8. SSM scan via custom autograd Function
class SSMScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, decay):
        B, T, H, dh = 1, mem.shape[0]//1, mem.shape[1]//(2*4), 4  # dummy, need real params
        # Actually need to pass B, T, H, dh explicitly
        raise NotImplementedError("Use the parameterized version")
    @staticmethod
    def backward(ctx, grad_output):
        pass

def grad_ssm_scan():
    B, T, H, dh = 1, 8, 2, 4
    # Use manual backward via PyTorch's autograd on the scan operation
    # Forward: ca[t] = cumprod(decay[0..t]), sf[t] = ca[t] * cumsum(mem[t]/(ca[t]+eps))
    # For vectorized forward:
    decay = torch.randn(B, T, H, device='cuda', dtype=torch.float64, requires_grad=True)
    mem = torch.randn(B, T, H, dh, device='cuda', dtype=torch.float64, requires_grad=True)
    decay_sig = torch.sigmoid(decay)
    
    ca = torch.cumprod(decay_sig, dim=1).clamp(min=1e-38)  # [B,T,H]
    ca_exp = ca.unsqueeze(-1).expand(-1, -1, -1, dh)  # [B,T,H,dh]
    wcs = torch.cumsum(mem / (ca_exp + 1e-8), dim=1)
    sf = ca_exp * wcs
    
    loss = sf.sum()
    loss.backward()
    
    save("ssm_scan",
         mem=to_np(mem.reshape(B*T, H*dh)),
         decay=to_np(decay.reshape(B*T, H)),
         dmem=to_np(mem.grad.reshape(B*T, H*dh)),
         ddecay=to_np(decay.grad.reshape(B*T, H)))

# 9. Sigmoid
def grad_sigmoid():
    x = d(256)
    out = torch.sigmoid(x)
    loss = out.sum()
    loss.backward()
    save("sigmoid", x=to_np(x), dx=to_np(x.grad))

# 10. Full 1-layer backward (compare weight gradients)
def grad_layer_ssm(cfg, state_dict):
    """Run one full forward+backward for an SSM layer using our model weights.
    Compare weight gradients with our engine."""
    # This requires loading the full model - done separately in C++
    pass

print("Generating gradcheck references...")
grad_silu()
grad_silu_mul()
grad_linear()
grad_crossentropy()
grad_layernorm()
grad_embedding()
grad_rope()
grad_sigmoid()
print("Done.")
