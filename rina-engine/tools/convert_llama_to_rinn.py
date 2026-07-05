#!/usr/bin/env python3
import struct, json, math, torch, numpy as np, sys, os
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(__file__))
from q4_pack import pack_q4_0, pack_q4_0_f, q4_0_storage_size, q4_0f_storage_size

_NON_QAT = ('input_layernorm', 'post_attention_layernorm', 'norm', 'ln1', 'ln2', 'ln_f')

def is_qat_key(key):
    if 'embed_tokens' in key or 'lm_head' in key or 'wte' in key:
        return False
    for skip in _NON_QAT:
        if skip in key:
            return False
    if '.weight' in key or '.bias' in key:
        return True
    return False

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

def convert(ckpt_dir, output_path, max_seq=512, qbits=32, q4f=False):
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
    print(f"Quantization: {'q4_0' if qbits == 4 else 'fp32'}")
    with safe_open(ckpt, framework="pt", device="cpu") as st:
        tensors = []
        def add(n, arr):
            is_q4 = (is_qat_key(n) and (qbits == 4 or qbits == '4f'))
            is_q4f = (q4f and is_q4)
            s = list(arr.shape) + [0] * (4 - arr.ndim)
            if is_q4:
                packer = pack_q4_0_f if is_q4f else pack_q4_0
                packed = packer(arr)
                size = q4_0f_storage_size if is_q4f else q4_0_storage_size
                size = size(int(np.prod(arr.shape)))
                assert len(packed) == size
                qt = 7 if is_q4f else 5
                tensors.append({"name": n, "shape": s, "quant_type": qt, "block_size": 32,
                                "size": size, "data": packed})
            else:
                b = arr.numpy().astype(np.float32).tobytes()
                tensors.append({"name": n, "shape": s, "quant_type": 6, "block_size": 1,
                                "size": len(b), "data": b})
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
        # lm_head.weight is weight-tied with wte; engine falls back to wte when missing
        # add("lm_head.weight", wte)

    idx_entry_size = lambda n: 2 + len(n.encode()) + 4*4 + 1 + 2 + 8 + 8
    idx_sz = sum(idx_entry_size(t["name"]) for t in tensors)
    data_off = ((512 + len(cj) + idx_sz + 255) // 256) * 256
    off = data_off
    for t in tensors: t["offset"] = off; off += t["size"]
    total_mb = off / 1024 / 1024
    print(f"Writing {output_path} ({len(tensors)} tensors, {total_mb:.0f} MB)...")
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
            f.write(struct.pack("<B", t["quant_type"]))
            f.write(struct.pack("<H", t["block_size"]))
            f.write(struct.pack("<Q", t["offset"])); f.write(struct.pack("<Q", t["size"]))
        pad = data_off - f.tell()
        if pad > 0: f.write(b"\x00" * pad)
        for t in tensors: f.write(t["data"])
    print(f"Done: {output_path}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/aquama/Development/RINA_Project/models/teacher/LLM-Research/Llama-3___2-1B-Instruct"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llama3.2-1b.rinn"
    qbits = 32
    for i, a in enumerate(sys.argv):
        if a == "--qbits" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1]
            qbits = int(v) if v.isdigit() else v
    convert(ckpt, out, qbits=qbits, q4f=(qbits == '4f'))
