"""Q4_0 block-wise quantization: pack/unpack utilities.

block_q4_0: {half scale; uint8_t data[16];} — 18 bytes per 32 values
Each value ∈ [-7, 7], stored as unsigned with +7 offset → [0, 14].
Packed 2 per byte: data[j] = (s[2j+1] << 4) | s[2j].
"""
import struct
import torch


def pack_q4_0(tensor: torch.Tensor) -> bytes:
    """Pack fp32 weight tensor to q4_0 block format bytes.
    
    tensor: (..., K) where last dim is multiple of 32.
    Returns: packed bytes in block_q4_0 format.
    """
    flat = tensor.reshape(-1, 32).float()
    amax = flat.abs().max(dim=-1, keepdim=True).values
    scale = amax / 7.0
    scale = scale.clamp(min=1e-10)

    q, scale = _quantize_blocks(flat, scale)
    return _pack_blocks_q4_0(q, scale)

def pack_q4_0_calibrated(tensor: torch.Tensor) -> bytes:
    """Pack with calibration-optimized per-block scales (vectorized).
    
    For each block, searches over candidate scales around max/7
    to find the one that minimizes dequantization MSE.
    Compatible with standard Q4_0 format — only the scale values change.
    """
    flat = tensor.reshape(-1, 32).float()  # [N, 32]
    amax = flat.abs().max(dim=-1, keepdim=True).values  # [N, 1]
    base_scale = (amax / 7.0).clamp(min=1e-10)

    # Candidates as fractions of base_scale
    cf = torch.tensor([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3],
                      device=flat.device, dtype=torch.float32)
    n_cand = len(cf)

    # Expand: [N, 32] → [N, C, 32], scales → [N, C, 1]
    flat_e = flat.unsqueeze(1).expand(-1, n_cand, -1)  # [N, C, 32]
    scale_e = (base_scale * cf.unsqueeze(0)).unsqueeze(-1)  # [N, C, 1]
    scaled = flat_e / scale_e                            # [N, C, 32]

    # Quantize
    sign = scaled.sign()
    q = (scaled.abs() + 0.5).floor().clamp(0, 7).int() * sign.int()
    q = q.clamp(-7, 7)
    qu = (q + 7).float()                                 # [N, C, 32]

    # Dequantize and compute MSE per block per candidate
    dq = (qu - 7) * scale_e                              # [N, C, 32]
    mse = ((flat_e - dq) ** 2).mean(dim=-1)              # [N, C]

    # Best candidate per block
    best_idx = mse.argmin(dim=-1)                        # [N]
    best_q_3d = qu.gather(1, best_idx.view(-1, 1, 1).expand(-1, -1, 32)).squeeze(1).int()
    best_s = scale_e.squeeze(-1).gather(1, best_idx.unsqueeze(-1))

    return _pack_blocks_q4_0(best_q_3d, best_s.unsqueeze(-1))

def _quantize(block, scale):
    """Quantize a [1, 32] block with given scale."""
    scaled = block / scale
    sign = scaled.sign()
    q = (scaled.abs() + 0.5).floor().clamp(0, 7).int() * sign.int()
    q = q.clamp(-7, 7)
    return q + 7

def _quantize_blocks(flat, scale):
    """Quantize all blocks with given scales."""
    scaled = flat / scale
    sign = scaled.sign()
    q = (scaled.abs() + 0.5).floor().clamp(0, 7).int() * sign.int()
    q = q.clamp(-7, 7)
    q_unsigned = q + 7
    return q_unsigned, scale

def _pack_blocks_q4_0(q_unsigned, scale):
    """Pack quantized blocks into Q4_0 format bytes."""
    q_unsigned = q_unsigned.int()
    even = q_unsigned[:, 0::2]
    odd = q_unsigned[:, 1::2]
    packed = (odd << 4) | even

    blocks = []
    for i in range(len(scale)):
        scale_bytes = struct.pack('<e', scale[i, 0].item())
        data_bytes = bytes(packed[i].tolist())
        blocks.append(scale_bytes + data_bytes)
    return b''.join(blocks)


def pack_q4_0_f(tensor: torch.Tensor) -> bytes:
    """Same as pack_q4_0 but stores scale as fp32 (20 bytes/block)."""
    flat = tensor.reshape(-1, 32).float()
    amax = flat.abs().max(dim=-1, keepdim=True).values
    scale = amax / 7.0
    scale = scale.clamp(min=1e-10)

    scaled = flat / scale
    sign = scaled.sign()
    q = (scaled.abs() + 0.5).floor().clamp(0, 7).int() * sign.int()
    q = q.clamp(-7, 7)
    q_unsigned = q + 7

    even = q_unsigned[:, 0::2]
    odd = q_unsigned[:, 1::2]
    packed = (odd << 4) | even

    blocks = []
    for i in range(len(scale)):
        scale_bytes = struct.pack('<f', scale[i, 0].item())
        data_bytes = bytes(packed[i].tolist())
        blocks.append(scale_bytes + data_bytes)

    return b''.join(blocks)


def q4_0f_storage_size(n_elems: int) -> int:
    num_blocks = (n_elems + 31) // 32
    return num_blocks * 20

def q4_0_storage_size(n_elems: int) -> int:
    """Number of bytes needed to store n_elems in q4_0 format."""
    num_blocks = (n_elems + 31) // 32
    return num_blocks * 18  # 18 bytes per block_q4_0
