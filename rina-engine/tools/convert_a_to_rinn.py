#!/usr/bin/env python3
"""Convert RINA L3X (pure MLA) checkpoint to .rinn format (pure fp32)."""
import os, struct, json, torch, numpy as np, sys, shutil

def gen_config():
    return {
        "name": "rina-l3x", "dim": 640, "n_layers": 16,
        "n_heads": 10, "n_kv_heads": 5, "head_dim": 64, "d_c": 160,
        "d_h_r": 32, "vocab_size": 128256, "block_size": 512,
        "layers": [{"type": "sparse_gather_fa"} for _ in range(16)]
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

    def make_rope_from_inv(inv_freq, max_pos=512):
        t = torch.arange(max_pos, dtype=torch.float32)
        f = torch.outer(t, inv_freq.float())
        return f.cos(), f.sin()

    # Global weights
    add_tensor(tensors, "transformer.wte.weight", state["transformer.wte.weight"])
    add_tensor(tensors, "transformer.ln_f.weight", state["transformer.ln_f.weight"])
    add_tensor(tensors, "lm_head.weight", state["lm_head.weight"] if "lm_head.weight" in state else state["transformer.wte.weight"])

    for l in range(16):
        prefix = f"transformer.h.{l}."
        # Shared
        add_tensor(tensors, f"{prefix}ln1.weight", state[f"{prefix}ln1.weight"])
        add_tensor(tensors, f"{prefix}ln2.weight", state[f"{prefix}ln2.weight"])
        add_tensor(tensors, f"{prefix}mlp.w1.weight", state[f"{prefix}mlp.w1.weight"])
        add_tensor(tensors, f"{prefix}mlp.w2.weight", state[f"{prefix}mlp.w2.weight"])
        add_tensor(tensors, f"{prefix}mlp.w3.weight", state[f"{prefix}mlp.w3.weight"])

        # MLA weights (path prefix matches engine convention)
        ap = f"{prefix}attn."
        add_tensor(tensors, f"{prefix}path.w_dqkv.weight", state[f"{ap}w_dqkv.weight"])
        add_tensor(tensors, f"{prefix}path.q_norm.weight", state[f"{ap}q_norm.weight"])
        add_tensor(tensors, f"{prefix}path.k_norm.weight", state[f"{ap}k_norm.weight"])
        add_tensor(tensors, f"{prefix}path.w_uq.weight", state[f"{ap}w_uq.weight"])
        add_tensor(tensors, f"{prefix}path.w_uk.weight", state[f"{ap}w_uk.weight"])
        add_tensor(tensors, f"{prefix}path.w_k2v.weight", state[f"{ap}w_k2v.weight"])
        add_tensor(tensors, f"{prefix}path.w_qr.weight", state[f"{ap}w_qr.weight"])
        add_tensor(tensors, f"{prefix}path.w_kr.weight", state[f"{ap}w_kr.weight"])
        add_tensor(tensors, f"{prefix}path.c_proj.weight", state[f"{ap}c_proj.weight"])

        # RoPE tables
        inv_q = state[f"{ap}rope_q.inv_freq"]
        inv_k = state[f"{ap}rope.inv_freq"]
        cos_q, sin_q = make_rope_from_inv(inv_q)
        cos_k, sin_k = make_rope_from_inv(inv_k)
        add_tensor(tensors, f"{prefix}path.rope_q.cos", cos_q)
        add_tensor(tensors, f"{prefix}path.rope_q.sin", sin_q)
        add_tensor(tensors, f"{prefix}path.rope.cos", cos_k)
        add_tensor(tensors, f"{prefix}path.rope.sin", sin_k)

    # Write .rinn (or hybrid directory)
    ies = lambda n: 2 + len(n.encode()) + 4*4 + 1 + 2 + 8 + 8
    idx_sz = sum(ies(t["name"]) for t in tensors)
    data_off = ((512 + len(cj) + idx_sz + 255) // 256) * 256
    off = data_off
    for t in tensors: t["offset"] = off; off += t["size"]

    # Check if output is hybrid directory: if path has no extension, treat as dir
    is_dir = '.' not in os.path.basename(output_path)
    if is_dir:
        os.makedirs(output_path, exist_ok=True)
        rinn_path = os.path.join(output_path, "weights.rinn")
    else:
        rinn_path = output_path
    
    print(f"Writing {rinn_path} ({len(tensors)} tensors, {off/1024/1024:.0f} MB)...")
    os.makedirs(os.path.dirname(rinn_path) or '.', exist_ok=True)
    with open(rinn_path, "wb") as f:
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
            for _ in range(4): f.write(struct.pack("<i", t["shape"][_]))
            f.write(struct.pack("<B", 6)); f.write(struct.pack("<H", 1))
            f.write(struct.pack("<Q", t["offset"])); f.write(struct.pack("<Q", t["size"]))
        pad = data_off - f.tell()
        if pad > 0: f.write(b"\x00" * pad)
        for t in tensors: f.write(t["data"])
    
    # Copy tokenizer if available (from HF checkpoint path or teacher model)
    # For L3X, look for tokenizer files alongside the checkpoint
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    teacher = os.path.join(os.path.dirname(ckpt_dir), "teacher", "LLM-Research", "Llama-3___2-1B-Instruct")
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(ckpt_dir, tok_file)
        if not os.path.exists(src):
            src = os.path.join(teacher, tok_file)
        if os.path.exists(src) and is_dir:
            import shutil; shutil.copy2(src, os.path.join(output_path, tok_file))
    
    if is_dir:
        # Also save config.json for hybrid format
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Done: {rinn_path}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/out-0.1b-a-v2/a_final.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/model_a.rinn"
    convert(ckpt, out)
