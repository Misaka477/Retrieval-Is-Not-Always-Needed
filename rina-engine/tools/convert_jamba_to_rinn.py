#!/usr/bin/env python3
"""Convert RINA Jamba QW2 checkpoint to .rinn format.

Supports fp32 (default) and q4_0 weight quantization (--qbits 4).
"""
import struct, json, torch, numpy as np, sys, os

# Add tools dir to path for q4_pack
sys.path.insert(0, os.path.dirname(__file__))
from q4_pack import pack_q4_0, q4_0_storage_size

# Weights NOT to quantize: norm, embedding, RoPE tables
_NON_QAT_SUFFIXES = ('q_norm.weight', 'k_norm.weight',
    'ln1.weight', 'ln2.weight', 'ln_f.weight',
    'rope_q.cos', 'rope_q.sin', 'rope.cos', 'rope.sin')


def is_qat_weight(name: str) -> bool:
    """True if this weight should be quantized (if --qbits is set)."""
    if name == 'transformer.wte.weight':
        return False  # embedding stays fp32
    if name == 'lm_head.weight':
        return False  # weight-tied with wte, stays fp32
    if any(name.endswith(s) for s in _NON_QAT_SUFFIXES):
        return False
    if name.endswith('.weight'):
        parts = name.split('.')
        if len(parts) >= 4:
            third = parts[3]
            if third in ('mlp', 'path'):
                return True
    return False


def gen_config():
    """Jamba QW2: SSM×3 + Sparse×4 repeated 4 times = 16 layers"""
    types = (["inertia_wave_ssm"]*3 + ["sparse_gather_fa"]) * 4
    return {
        "name": "rina-jamba-qw2", "dim": 640, "n_layers": 16,
        "n_heads": 10, "n_kv_heads": 5, "head_dim": 64, "d_c": 160,
        "d_h_r": 32, "vocab_size": 128256, "block_size": 512,
        "ssm_steps": 3, "quant_mode": "", "ssm_qbits": 0,
        "layers": [{"type": t} for t in types]
    }


def add_tensor(tensors, name, arr, qbits=32):
    """Add a tensor to the list. If qbits=4 and weight is QAT, quantize to q4_0."""
    is_q4 = (qbits == 4 and is_qat_weight(name))
    n_elems = int(np.prod(arr.shape))

    if is_q4:
        packed = pack_q4_0(arr)
        size = q4_0_storage_size(n_elems)
        assert len(packed) == size, f"pack size mismatch: {len(packed)} vs {size}"
        s = list(arr.shape) + [0] * (4 - len(arr.shape))
        tensors.append({
            "name": name, "shape": s,
            "quant_type": 5, "block_size": 32,
            "size": size, "data": packed
        })
    else:
        b = arr.numpy().astype(np.float32).tobytes()
        s = list(arr.shape) + [0] * (4 - len(arr.shape))
        tensors.append({
            "name": name, "shape": s,
            "quant_type": 6, "block_size": 1,
            "size": len(b), "data": b
        })


def convert(ckpt_path, output_path, qbits=32):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = sd['model'] if 'model' in sd else sd
    config = gen_config()
    cj = json.dumps(config).encode("utf-8")
    tensors = []

    def make_rope_from_inv(inv_freq, max_pos=512):
        t = torch.arange(max_pos, dtype=torch.float32)
        f = torch.outer(t, inv_freq.float())
        return f.cos(), f.sin()

    print(f"Quantization: {'q4_0 (QAT weights)' if qbits == 4 else 'fp32'}")
    add_tensor(tensors, "transformer.wte.weight",
               state["transformer.wte.weight"], qbits)
    add_tensor(tensors, "transformer.ln_f.weight",
               state["transformer.ln_f.weight"], qbits)
    lm_head = state.get("lm_head.weight",
                        state["transformer.wte.weight"])
    # lm_head.weight is weight-tied with wte; engine falls back to wte when missing
    # add_tensor(tensors, "lm_head.weight", lm_head, qbits)

    for l in range(16):
        prefix = f"transformer.h.{l}."
        add_tensor(tensors, f"{prefix}ln1.weight",
                   state[f"{prefix}ln1.weight"], qbits)
        add_tensor(tensors, f"{prefix}ln2.weight",
                   state[f"{prefix}ln2.weight"], qbits)
        add_tensor(tensors, f"{prefix}mlp.w1.weight",
                   state[f"{prefix}mlp.w1.weight"], qbits)
        add_tensor(tensors, f"{prefix}mlp.w2.weight",
                   state[f"{prefix}mlp.w2.weight"], qbits)
        add_tensor(tensors, f"{prefix}mlp.w3.weight",
                   state[f"{prefix}mlp.w3.weight"], qbits)
        add_tensor(tensors, f"{prefix}path.q_norm.weight",
                   state[f"{prefix}path.q_norm.weight"], qbits)

        if f"{prefix}path.w_dq.weight" in state:
            # SSM layer
            add_tensor(tensors, f"{prefix}path.w_dq.weight",
                       state[f"{prefix}path.w_dq.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_out.weight",
                       state[f"{prefix}path.w_out.weight"], qbits)
            for k in range(3):
                add_tensor(tensors, f"{prefix}path.w_mem.{k}.weight",
                           state[f"{prefix}path.w_mem.{k}.weight"], qbits)
                add_tensor(tensors, f"{prefix}path.w_decay.{k}.weight",
                           state[f"{prefix}path.w_decay.{k}.weight"], qbits)
            cos_z = torch.zeros(512, 1)
            sin_z = torch.zeros(512, 1)
            add_tensor(tensors, f"{prefix}path.rope_q.cos", cos_z, qbits)
            add_tensor(tensors, f"{prefix}path.rope_q.sin", sin_z, qbits)
            add_tensor(tensors, f"{prefix}path.rope.cos", cos_z, qbits)
            add_tensor(tensors, f"{prefix}path.rope.sin", sin_z, qbits)
        else:
            # MLA (sparse attention) layer
            add_tensor(tensors, f"{prefix}path.w_dqkv.weight",
                       state[f"{prefix}path.w_dqkv.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_uq.weight",
                       state[f"{prefix}path.w_uq.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_uk.weight",
                       state[f"{prefix}path.w_uk.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_k2v.weight",
                       state[f"{prefix}path.w_k2v.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_qr.weight",
                       state[f"{prefix}path.w_qr.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.w_kr.weight",
                       state[f"{prefix}path.w_kr.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.c_proj.weight",
                       state[f"{prefix}path.c_proj.weight"], qbits)
            add_tensor(tensors, f"{prefix}path.k_norm.weight",
                       state[f"{prefix}path.k_norm.weight"], qbits)
            inv_q = state[f"{prefix}path.rope_q.inv_freq"]
            inv_k = state[f"{prefix}path.rope.inv_freq"]
            cos_q, sin_q = make_rope_from_inv(inv_q)
            cos_k, sin_k = make_rope_from_inv(inv_k)
            add_tensor(tensors, f"{prefix}path.rope_q.cos", cos_q, qbits)
            add_tensor(tensors, f"{prefix}path.rope_q.sin", sin_q, qbits)
            add_tensor(tensors, f"{prefix}path.rope.cos", cos_k, qbits)
            add_tensor(tensors, f"{prefix}path.rope.sin", sin_k, qbits)

    # Write .rinn
    idx_entry_size = lambda n: 2 + len(n.encode()) + 4*4 + 1 + 2 + 8 + 8
    idx_sz = sum(idx_entry_size(t["name"]) for t in tensors)
    data_off = ((512 + len(cj) + idx_sz + 255) // 256) * 256
    off = data_off
    for t in tensors:
        t["offset"] = off
        off += t["size"]

    total_mb = off / 1024 / 1024
    print(f"Writing {output_path} ({len(tensors)} tensors, {total_mb:.0f} MB)...")
    with open(output_path, "wb") as f:
        hdr = bytearray(512)
        hdr[0:4] = b"RINN"
        struct.pack_into("<I", hdr, 4, 1)
        struct.pack_into("<I", hdr, 8, len(tensors))
        struct.pack_into("<Q", hdr, 16, 512)
        struct.pack_into("<Q", hdr, 24, len(cj))
        struct.pack_into("<Q", hdr, 32, 512 + len(cj))
        struct.pack_into("<Q", hdr, 40, idx_sz)
        struct.pack_into("<Q", hdr, 48, data_off)
        f.write(hdr)
        f.write(cj)
        for t in tensors:
            nb = t["name"].encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            for d_ in range(4):
                f.write(struct.pack("<i", t["shape"][d_]))
            f.write(struct.pack("<B", t["quant_type"]))
            f.write(struct.pack("<H", t["block_size"]))
            f.write(struct.pack("<Q", t["offset"]))
            f.write(struct.pack("<Q", t["size"]))
        pad = data_off - f.tell()
        if pad > 0:
            f.write(b"\x00" * pad)
        for t in tensors:
            f.write(t["data"])
    print(f"Done: {output_path}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "models/out-rina-jamba-qw2/jambaqw2_final.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/jamba_qw2.rinn"
    qbits = 32
    for i, a in enumerate(sys.argv):
        if a == "--qbits" and i + 1 < len(sys.argv):
            qbits = int(sys.argv[i + 1])
    convert(ckpt, out, qbits)
