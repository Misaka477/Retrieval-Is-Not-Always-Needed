#!/usr/bin/env python3
"""
load_hf.py — HuggingFace safetensors → RINN model converter.

Detects model architecture from config.json, maps HF weight names to
RINN format, quantizes if requested, and writes .rinn file.

Usage:
  python3 load_hf.py <hf_model_path> <output.rinn> [--qbits 4|4f|32]
"""
import struct, json, math, torch, numpy as np, sys, os
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(__file__))
from q4_pack import pack_q4_0, pack_q4_0_f, pack_q4_0_calibrated, q4_0_storage_size, q4_0f_storage_size

# ── Tensor types in rinn format ──────────────────────────────────
QT_FP32 = 6
QT_Q4_0 = 5
QT_Q4_0F = 7

# Layers that should NEVER be quantized
_NON_QAT_KEYS = ('input_layernorm', 'post_attention_layernorm',
                 'norm', 'ln1', 'ln2', 'ln_f', 'embed_tokens', 'lm_head',
                 'wte')

def should_quant(name):
    if 'wte' in name or 'lm_head' in name:
        return False
    for skip in _NON_QAT_KEYS:
        if skip in name:
            return False
    return '.weight' in name

# ── Rope ─────────────────────────────────────────────────────────
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

# ── Architecture registry ────────────────────────────────────────

class ArchMapper:
    """Maps HF model architecture to RINN weight names and config."""
    def model_type(self): raise NotImplementedError
    def rina_name(self): raise NotImplementedError
    def rina_head_dim(self, cfg): return cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    def rina_d_h_r(self, cfg): return self.rina_head_dim(cfg)
    def rina_d_c(self, cfg): return 0
    def weight_map(self, layer, hf_name):
        """Map HF weight name at layer `layer` to RINN name, or None if not applicable."""
        raise NotImplementedError
    def global_weight_map(self):
        """Map non-layer weights (embed, norm) to RINN names."""
        return {}

class LlamaMapper(ArchMapper):
    def model_type(self): return "llama"
    def rina_name(self): return "llama"
    def weight_map(self, l, hf_name):
        m = {
            "input_layernorm.weight":      f"transformer.h.{l}.ln1.weight",
            "post_attention_layernorm.weight": f"transformer.h.{l}.ln2.weight",
            "self_attn.q_proj.weight":     f"transformer.h.{l}.attn.w_q.weight",
            "self_attn.k_proj.weight":     f"transformer.h.{l}.attn.w_k.weight",
            "self_attn.v_proj.weight":     f"transformer.h.{l}.attn.w_v.weight",
            "self_attn.o_proj.weight":     f"transformer.h.{l}.attn.w_o.weight",
            "mlp.gate_proj.weight":        f"transformer.h.{l}.mlp.w1.weight",
            "mlp.up_proj.weight":          f"transformer.h.{l}.mlp.w3.weight",
            "mlp.down_proj.weight":        f"transformer.h.{l}.mlp.w2.weight",
        }
        return m.get(hf_name)
    def global_weight_map(self):
        return {
            "model.embed_tokens.weight": "transformer.wte.weight",
            "model.norm.weight":         "transformer.ln_f.weight",
        }

class MistralMapper(LlamaMapper):
    def model_type(self): return "mistral"
    def rina_name(self): return "mistral"

class Qwen2Mapper(LlamaMapper):
    def model_type(self): return "qwen2"
    def rina_name(self): return "qwen2"

class GemmaMapper(ArchMapper):
    def model_type(self): return "gemma"
    def rina_name(self): return "gemma"
    def rina_d_h_r(self, cfg): return cfg.get("head_dim", 256)
    def weight_map(self, l, hf_name):
        # Gemma uses different MLP naming (gate_fc vs gate_proj)
        m = {
            "input_layernorm.weight":     f"transformer.h.{l}.ln1.weight",
            "post_attention_layernorm.weight": f"transformer.h.{l}.ln2.weight",
            "self_attn.q_proj.weight":    f"transformer.h.{l}.attn.w_q.weight",
            "self_attn.k_proj.weight":    f"transformer.h.{l}.attn.w_k.weight",
            "self_attn.v_proj.weight":    f"transformer.h.{l}.attn.w_v.weight",
            "self_attn.o_proj.weight":    f"transformer.h.{l}.attn.w_o.weight",
            "mlp.gate_proj.weight":       f"transformer.h.{l}.mlp.w1.weight",
            "mlp.up_proj.weight":         f"transformer.h.{l}.mlp.w3.weight",
            "mlp.down_proj.weight":       f"transformer.h.{l}.mlp.w2.weight",
        }
        return m.get(hf_name)
    def global_weight_map(self):
        return {
            "model.embed_tokens.weight": "transformer.wte.weight",
            "model.norm.weight":         "transformer.ln_f.weight",
        }

# Register architectures
_ARCH_MAPPERS = [cls() for cls in ArchMapper.__subclasses__()]

def detect_arch(cfg) -> ArchMapper:
    mt = cfg.get("model_type", "")
    for m in _ARCH_MAPPERS:
        if m.model_type() == mt:
            return m
    # Default to Llama mapper for compatible architectures
    if any(k in cfg.get("architectures", []) for k in ["LlamaForCausalLM", "MistralForCausalLM"]):
        return LlamaMapper()
    raise ValueError(f"Unknown model_type: {mt}. Supported: {[m.model_type() for m in _ARCH_MAPPERS]}")

# ── Model loading ────────────────────────────────────────────────

def find_safetensors(ckpt_dir):
    """Find the first .safetensors file in the directory (prefer model.safetensors or index)."""
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        return os.path.join(ckpt_dir, idx.get("weight_map", {}).get(list(idx.get("weight_map", {}))[0], ""))
    # Direct single file
    for fname in ["model.safetensors", "consolidated.safetensors"]:
        p = os.path.join(ckpt_dir, fname)
        if os.path.exists(p):
            return p
    # Any safetensors
    import glob
    files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
    if files:
        return files[0]
    raise FileNotFoundError(f"No .safetensors found in {ckpt_dir}")

def load_model(ckpt_dir, output_path, max_seq=512, qbits=32, q4f=False, calibrated=False):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)

    arch = detect_arch(hf_cfg)
    d = hf_cfg["hidden_size"]
    n_layers = hf_cfg["num_hidden_layers"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv_heads = hf_cfg.get("num_key_value_heads", n_heads)
    head_dim = arch.rina_head_dim(hf_cfg)
    d_h_r = arch.rina_d_h_r(hf_cfg)
    d_c = arch.rina_d_c(hf_cfg)
    V = hf_cfg["vocab_size"]
    rope_t = hf_cfg.get("rope_theta", 10000.0)
    rope_s = hf_cfg.get("rope_scaling")
    weight_tying = hf_cfg.get("tie_word_embeddings", True)

    rina_cfg = {
        "name": f"{arch.rina_name()}-{d//100_000_000:.1f}b" if d >= 100_000_000 else f"{arch.rina_name()}-{d}dim",
        "dim": d, "n_layers": n_layers,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "d_c": d_c, "d_h_r": d_h_r,
        "vocab_size": V, "block_size": max_seq,
        "ssm_steps": 0, "weight_tying": weight_tying,
        "quant_mode": "", "ssm_qbits": 0,
        "layers": [{"type": "standard_attention"} for _ in range(n_layers)]
    }
    cj = json.dumps(rina_cfg).encode("utf-8")
    cos, sin = make_rope(d_h_r, max_seq, rope_t, rope_s, "cuda")

    ckpt = find_safetensors(ckpt_dir)
    print(f"Model: {arch.rina_name()} ({d}dim, {n_layers}L, {n_heads}H, {n_kv_heads}KV)")
    print(f"Loading: {ckpt}")
    print(f"Quantization: {qbits} bit{' ('+('q4f' if q4f else 'q4_0')+')' if qbits != 32 else 'fp32'}")

    with safe_open(ckpt, framework="pt", device="cpu") as st:
        # Collect all keys
        all_keys = list(st.keys())
        # Build HF→RINN layer mapping
        hf_to_rinn = {}
        for l in range(n_layers):
            for hf_name in all_keys:
                key_prefix = f"model.layers.{l}."
                if not hf_name.startswith(key_prefix):
                    continue
                hf_local = hf_name[len(key_prefix):]
                rn = arch.weight_map(l, hf_local)
                if rn:
                    hf_to_rinn[hf_name] = rn
        # Global weights
        for hf_name, rn in arch.global_weight_map().items():
            if hf_name in all_keys:
                hf_to_rinn[hf_name] = rn

        # Read and pack tensors
        tensors = []
        def add(n, arr):
            q = (should_quant(n) and (qbits == 4 or qbits == '4f'))
            q4f_actual = (q4f and q)
            s = list(arr.shape) + [0] * (4 - arr.ndim)
            if q:
                if calibrated:
                    packer = pack_q4_0_calibrated
                else:
                    packer = pack_q4_0_f if q4f_actual else pack_q4_0
                packed = packer(arr)
                size_fn = q4_0f_storage_size if q4f_actual else q4_0_storage_size
                sz = size_fn(int(np.prod(arr.shape)))
                assert len(packed) == sz
                qt = QT_Q4_0F if q4f_actual else QT_Q4_0
                tensors.append({"name": n, "shape": s, "quant_type": qt,
                                "block_size": 32, "size": sz, "data": packed})
            else:
                b = arr.numpy().astype(np.float32).tobytes()
                tensors.append({"name": n, "shape": s, "quant_type": QT_FP32,
                                "block_size": 1, "size": len(b), "data": b})

        # Process layers
        for l in range(n_layers):
            if l % 4 == 0:
                print(f"  layer {l}/{n_layers}...")
            add(f"transformer.h.{l}.attn.rope_q.cos", cos.cpu())
            add(f"transformer.h.{l}.attn.rope_q.sin", sin.cpu())
            add(f"transformer.h.{l}.attn.rope.cos", cos.cpu())
            add(f"transformer.h.{l}.attn.rope.sin", sin.cpu())
            for hf_name, rn in hf_to_rinn.items():
                if f"model.layers.{l}." in hf_name:
                    add(rn, st.get_tensor(hf_name).float())

        # Global weights
        print("  global weights...")
        for hf_name, rn in hf_to_rinn.items():
            if not any(f"model.layers.{l}." in hf_name for l in range(n_layers)):
                add(rn, st.get_tensor(hf_name).float())

    # Write to directory: output_path/weights.rinn + tokenizer files
    os.makedirs(output_path, exist_ok=True)

    idx_entry_size = lambda n: 2 + len(n.encode()) + 4*4 + 1 + 2 + 8 + 8
    idx_sz = sum(idx_entry_size(t["name"]) for t in tensors)
    data_off = ((512 + len(cj) + idx_sz + 255) // 256) * 256
    off = data_off
    for t in tensors:
        t["offset"] = off
        off += t["size"]
    total_mb = off / 1024 / 1024
    weights_path = os.path.join(output_path, "weights.rinn")
    print(f"Writing {weights_path} ({len(tensors)} tensors, {total_mb:.0f} MB)...")
    with open(weights_path, "wb") as f:
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

    # Copy tokenizer files from HF model dir
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "special_tokens_map.json", "added_tokens.json"]:
        src = os.path.join(ckpt_dir, tok_file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(output_path, tok_file))
            print(f"  copied {tok_file}")

    print(f"Done: {output_path}")

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 load_hf.py <hf_model_dir> <output_dir> [--qbits 4|4f|32]")
        print("  Converts HF model to RINN directory format (output_dir/weights.rinn + tokenizer)")
        sys.exit(1)
    ckpt = sys.argv[1]
    out = sys.argv[2]
    qbits = 32
    q4f = False
    calibrated = False
    for i, a in enumerate(sys.argv):
        if a == "--qbits" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1]
            if v == "4f":
                qbits = 4
                q4f = True
            elif v == "cal":
                qbits = 4
                calibrated = True
            else:
                qbits = int(v) if v.isdigit() else v
        if a == "--calibrated":
            calibrated = True
    load_model(ckpt, out, qbits=qbits, q4f=q4f, calibrated=calibrated)
