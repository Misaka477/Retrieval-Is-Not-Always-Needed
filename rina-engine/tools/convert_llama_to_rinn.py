#!/usr/bin/env python3
import struct, json, math, torch, numpy as np, sys, os
from safetensors import safe_open

def make_rope(dim, max_pos, theta=500000.0, scaling=None, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    if scaling and scaling.get("rope_type") == "llama3":
        factor = scaling["factor"]; lf = scaling["low_freq_factor"]
        hf = scaling["high_freq_factor"]; om = scaling["original_max_position_embeddings"]
        lw = om / lf; hw = om / hf
        wl = 2 * math.pi / inv_freq
        inv = torch.where(wl > lw, inv_freq / factor, inv_freq)
        smooth = (om / wl - lf) / (hf - lf)
        smoothed = (1 - smooth) * inv / factor + smooth * inv
        mid = ~(wl < hw) * ~(wl > lw)
        inv = torch.where(mid, smoothed, inv)
        inv_freq = inv
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    eps = torch.outer(t, inv_freq)
    return eps.cos().contiguous(), eps.sin().contiguous()

def convert(ckpt_dir, output_path, max_seq=512):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f: hf_cfg = json.load(f)
    d = hf_cfg["hidden_size"]; n_layers = hf_cfg["num_hidden_layers"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv_heads = hf_cfg.get("num_key_value_heads", n_heads)
    head_dim = hf_cfg.get("head_dim", d // n_heads)
    V = hf_cfg["vocab_size"]; rope_t = hf_cfg.get("rope_theta", 500000.0)
    rope_s = hf_cfg.get("rope_scaling")
    rina_cfg = {
        "name": "llama3.2-1b", "dim": d, "n_layers": n_layers,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "d_c": 0, "d_h_r": head_dim, "vocab_size": V, "block_size": max_seq,
        "ssm_steps": 0, "weight_tying": True,
        "quant_mode": "", "ssm_qbits": 0,
        "layers": [{"type": "standard_attention"} for _ in range(n_layers)]
    }
    cj = json.dumps(rina_cfg).encode("utf-8")
    cos, sin = make_rope(head_dim, max_seq, rope_t, rope_s, "cuda")
    ckpt = os.path.join(ckpt_dir, "model.safetensors")
    print(f"Opening {ckpt}...")
    with safe_open(ckpt, framework="pt", device="cpu") as st:
        tensors = []
        def add(n, arr):
            b = arr.numpy().astype(np.float32).tobytes()
            s = list(arr.shape) + [0]*(4-arr.ndim)
            tensors.append({"name": n, "shape": s, "size": len(b), "data": b})
        for l in range(n_layers):
            if l % 4 == 0: print(f"  layer {l}/{n_layers}...")
            add(f"transformer.h.{l}.attn.rope_q.cos", cos.cpu())
            add(f"transformer.h.{l}.attn.rope_q.sin", sin.cpu())
            add(f"transformer.h.{l}.attn.rope.cos", cos.cpu())
            add(f"transformer.h.{l}.attn.rope.sin", sin.cpu())
            for nm, rn in [("input_layernorm","ln1.weight"),("post_attention_layernorm","ln2.weight"),
                           ("self_attn.q_proj","attn.w_q.weight"),("self_attn.k_proj","attn.w_k.weight"),
                           ("self_attn.v_proj","attn.w_v.weight"),("self_attn.o_proj","attn.w_o.weight"),
                           ("mlp.gate_proj","mlp.w1.weight"),("mlp.up_proj","mlp.w3.weight"),
                           ("mlp.down_proj","mlp.w2.weight")]:
                t = st.get_tensor(f"model.layers.{l}.{nm}.weight").float()
                add(f"transformer.h.{l}.{rn}", t)
        print("  embedding + final norm...")
        wte = st.get_tensor("model.embed_tokens.weight").float()
        add("transformer.wte.weight", wte)
        add("transformer.ln_f.weight", st.get_tensor("model.norm.weight").float())
        add("lm_head.weight", wte)
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
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/aquama/Development/RINA_Project/models/teacher/LLM-Research/Llama-3___2-1B-Instruct"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llama3.2-1b.rinn"
    convert(ckpt, out)
