#!/usr/bin/env python3
"""Convert RINA Jamba QW2 checkpoint to .rinn format (pure fp32)."""
import struct, json, torch, numpy as np, sys

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

def add_tensor(tensors, name, arr):
    b = arr.numpy().astype(np.float32).tobytes()
    s = list(arr.shape) + [0]*(4-len(arr.shape))
    tensors.append({"name": name, "shape": s, "size": len(b), "data": b})

def convert(ckpt_path, output_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = sd['model'] if 'model' in sd else sd
    config = gen_config()
    cj = json.dumps(config).encode("utf-8")
    tensors = []

    # Generate RoPE tables from inv_freq (stored in checkpoint)
    def make_rope_from_inv(inv_freq, max_pos=512):
        t = torch.arange(max_pos, dtype=torch.float32)
        f = torch.outer(t, inv_freq.float())
        return f.cos(), f.sin()

    # Embedding + final norm + lm_head
    add_tensor(tensors, "transformer.wte.weight", state["transformer.wte.weight"])
    add_tensor(tensors, "transformer.ln_f.weight", state["transformer.ln_f.weight"])
    add_tensor(tensors, "lm_head.weight", state["lm_head.weight"] if "lm_head.weight" in state else state["transformer.wte.weight"])

    for l in range(16):
        prefix = f"transformer.h.{l}."
        # LN + MLP (all layers have these)
        add_tensor(tensors, f"{prefix}ln1.weight", state[f"{prefix}ln1.weight"])
        add_tensor(tensors, f"{prefix}ln2.weight", state[f"{prefix}ln2.weight"])
        add_tensor(tensors, f"{prefix}mlp.w1.weight", state[f"{prefix}mlp.w1.weight"])
        add_tensor(tensors, f"{prefix}mlp.w2.weight", state[f"{prefix}mlp.w2.weight"])
        add_tensor(tensors, f"{prefix}mlp.w3.weight", state[f"{prefix}mlp.w3.weight"])
        add_tensor(tensors, f"{prefix}path.q_norm.weight", state[f"{prefix}path.q_norm.weight"])

        if f"{prefix}path.w_dq.weight" in state:
            # SSM layer
            add_tensor(tensors, f"{prefix}path.w_dq.weight", state[f"{prefix}path.w_dq.weight"])
            add_tensor(tensors, f"{prefix}path.w_out.weight", state[f"{prefix}path.w_out.weight"])
            for k in range(3):
                add_tensor(tensors, f"{prefix}path.w_mem.{k}.weight", state[f"{prefix}path.w_mem.{k}.weight"])
                add_tensor(tensors, f"{prefix}path.w_decay.{k}.weight", state[f"{prefix}path.w_decay.{k}.weight"])
            # Generate RoPE tables for SSM (0-dim)
            # SSM doesn't use RoPE directly (uses q_norm only)
            cos_z = torch.zeros(512, 1); sin_z = torch.zeros(512, 1)
            add_tensor(tensors, f"{prefix}path.rope_q.cos", cos_z)
            add_tensor(tensors, f"{prefix}path.rope_q.sin", sin_z)
            add_tensor(tensors, f"{prefix}path.rope.cos", cos_z)
            add_tensor(tensors, f"{prefix}path.rope.sin", sin_z)
        else:
            # MLA (sparse attention) layer
            add_tensor(tensors, f"{prefix}path.w_dqkv.weight", state[f"{prefix}path.w_dqkv.weight"])
            add_tensor(tensors, f"{prefix}path.w_uq.weight", state[f"{prefix}path.w_uq.weight"])
            add_tensor(tensors, f"{prefix}path.w_uk.weight", state[f"{prefix}path.w_uk.weight"])
            add_tensor(tensors, f"{prefix}path.w_k2v.weight", state[f"{prefix}path.w_k2v.weight"])
            add_tensor(tensors, f"{prefix}path.w_qr.weight", state[f"{prefix}path.w_qr.weight"])
            add_tensor(tensors, f"{prefix}path.w_kr.weight", state[f"{prefix}path.w_kr.weight"])
            add_tensor(tensors, f"{prefix}path.c_proj.weight", state[f"{prefix}path.c_proj.weight"])
            # k_norm (our own MLA variant has it)
            add_tensor(tensors, f"{prefix}path.k_norm.weight", state[f"{prefix}path.k_norm.weight"])
            # RoPE tables from inv_freq
            inv_q = state[f"{prefix}path.rope_q.inv_freq"]
            inv_k = state[f"{prefix}path.rope.inv_freq"]
            cos_q, sin_q = make_rope_from_inv(inv_q)
            cos_k, sin_k = make_rope_from_inv(inv_k)
            add_tensor(tensors, f"{prefix}path.rope_q.cos", cos_q)
            add_tensor(tensors, f"{prefix}path.rope_q.sin", sin_q)
            add_tensor(tensors, f"{prefix}path.rope.cos", cos_k)
            add_tensor(tensors, f"{prefix}path.rope.sin", sin_k)

    # Write .rinn
    ies = lambda n: 2 + len(n.encode()) + 4*4 + 1 + 2 + 8 + 8
    idx_sz = sum(ies(t["name"]) for t in tensors)
    data_off = ((512 + len(cj) + idx_sz + 255) // 256) * 256
    off = data_off
    for t in tensors: t["offset"] = off; off += t["size"]

    print(f"Writing {output_path} ({len(tensors)} tensors, {off/1024/1024:.0f} MB)...")
    with open(output_path, "wb") as f:
        hdr = bytearray(512)
        hdr[0:4] = b"RINN"; struct.pack_into("<I", hdr, 4, 1)
        struct.pack_into("<I", hdr, 8, len(tensors))
        struct.pack_into("<Q", hdr, 16, 512)
        struct.pack_into("<Q", hdr, 24, len(cj))
        struct.pack_into("<Q", hdr, 32, 512 + len(cj))
        struct.pack_into("<Q", hdr, 40, idx_sz)
        struct.pack_into("<Q", hdr, 48, data_off)
        f.write(hdr); f.write(cj)
        for t in tensors:
            nb = t["name"].encode()
            f.write(struct.pack("<H", len(nb))); f.write(nb)
            for d_ in range(4): f.write(struct.pack("<i", t["shape"][d_]))
            f.write(struct.pack("<B", 6)); f.write(struct.pack("<H", 1))
            f.write(struct.pack("<Q", t["offset"])); f.write(struct.pack("<Q", t["size"]))
        pad = data_off - f.tell()
        if pad > 0: f.write(b"\x00" * pad)
        for t in tensors: f.write(t["data"])
    print(f"Done: {output_path}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/out-rina-jamba-qw2/jambaqw2_final.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/jamba_qw2.rinn"
    convert(ckpt, out)
