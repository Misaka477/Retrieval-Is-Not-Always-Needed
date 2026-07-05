#!/usr/bin/env python3
"""
calibrate.py — Per-channel activation statistics + SmoothQuant scales
for improved q4 quantization of RINN models.

Captures layer-wise activation distributions through a PyTorch model,
computes per-channel statistics and SmoothQuant smoothing factors,
outputs a .npz file consumed by load_hf.py.

Usage:
  python3 calibrate.py <hf_model_dir> <output_dir> [--calib-data <npy>] [--n-samples 128]
"""
import os, sys, json, torch, math, numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def get_calib_data(data_path, n_samples=64, seq_len=512):
    """Load token IDs from npy, extract n_samples sequences of seq_len tokens."""
    data = np.load(data_path, mmap_mode='r')
    total = len(data)
    usable = total - seq_len
    n = min(n_samples, usable)
    rng = np.random.RandomState(42)
    starts = rng.randint(0, usable, size=n)
    batch = np.stack([data[s:s+seq_len].astype(np.int64) for s in starts])
    print(f"Calibration data: {batch.shape} seqs × {seq_len} tokens, range [{data.min()},{data.max()}]")
    return batch

def register_hooks(model):
    """Register forward hooks to capture per-channel input activations
    for all linear layers."""
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach().float()  # [B, T, d] or [B*T, d]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            # Per-channel: mean abs, max abs, mean, std
            abs_x = x.abs()
            stats = {
                'mean_abs': abs_x.mean(dim=0).cpu().numpy(),  # [d]
                'max_abs':  abs_x.max(dim=0).values.cpu().numpy(),
                'mean':     x.mean(dim=0).cpu().numpy(),
                'std':      x.std(dim=0).cpu().numpy(),
            }
            if name not in activations:
                activations[name] = []
            activations[name].append(stats)
        return hook

    def install(root, prefix=""):
        for name, child in root.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                h = child.register_forward_hook(make_hook(full))
                handles.append(h)
            install(child, full)

    install(model)
    return activations, handles

def compute_scales(activations, n_layers, hidden_size):
    """Aggregate activation stats and compute per-channel SmoothQuant scales."""
    scales = {}
    for name, stats_list in activations.items():
        # Aggregate over all calibration batches
        mean_abs = np.stack([s['mean_abs'] for s in stats_list]).mean(axis=0)
        max_abs  = np.stack([s['max_abs'] for s in stats_list]).max(axis=0)
        means    = np.stack([s['mean'] for s in stats_list]).mean(axis=0)
        stds     = np.sqrt(np.stack([s['std']**2 for s in stats_list]).mean(axis=0))

        scales[name] = {
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'mean': means,
            'std': stds,
        }

    # Compute SmoothQuant per-channel smoothing factors
    # s_j = max(|X_j|)^α (for each input channel of each linear layer)
    # The weight will be multiplied by s_j, so the corresponding
    # activation channel becomes X_j / s_j (fused into preceding op).
    # α=0.5 balances between weight and activation difficulty.
    smooth_factors = {}
    alpha = 0.5
    for name, stats in scales.items():
        max_abs = stats['max_abs']
        # SmoothQuant scale per input channel
        smooth_factors[name] = (max_abs / max_abs.max()) ** alpha

    return scales, smooth_factors

def calibrate(model_path, output_dir, calib_data_path, n_samples=128, seq_len=2048):
    from transformers import AutoModelForCausalLM

    # Use bf16 on GPU if available (saves ~50% memory vs fp32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading model ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
        attn_implementation="eager"
    ).eval()
    cfg = model.config
    hidden_size = cfg.hidden_size
    n_layers = cfg.num_hidden_layers

    # Load calibration data
    calib_ids = get_calib_data(calib_data_path, n_samples, seq_len)
    calib_t = torch.tensor(calib_ids, device=device)

    # Register hooks
    activations, handles = register_hooks(model)
    # Process 1-2 at a time to avoid OOM
    chunk_size = 2

    print(f"Running {n_samples} calibration samples (chunks of {chunk_size})...")
    for i in range(0, n_samples, chunk_size):
        chunk = calib_t[i:i+chunk_size]
        with torch.no_grad():
            model(chunk)
        if (i + chunk_size) % 32 == 0 or i + chunk_size >= n_samples:
            print(f"  processed {min(i+chunk_size, n_samples)}/{n_samples}")

    # Remove hooks
    for h in handles:
        h.remove()

    print("Computing scales...")
    scales, smooth_factors = compute_scales(activations, n_layers, hidden_size)

    # Output to a directory with enough space
    real_dir = output_dir
    # Auto-redirect if tmp is too small
    import shutil
    if output_dir.startswith('/tmp') and not os.path.exists(output_dir):
        try:
            st = shutil.disk_usage('/tmp')
            if st.free < 200_000_000:  # <200MB free
                real_dir = os.path.expanduser(f"~/calib_out_{os.path.basename(output_dir)}")
                print(f"  /tmp low on space, using {real_dir}")
                output_dir = real_dir
        except: pass

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "calib_data.npz")

    # Save as npz (use structured arrays for JSON-like access)
    npz_data = {}
    for name, stats in scales.items():
        # Map internal HF name to RINN name
        rinn_name = hf_to_rinn_name(name, n_layers)
        if rinn_name:
            for k, v in stats.items():
                npz_data[f"{rinn_name}.{k}"] = v
    # Save smooth factors
    for name, factors in smooth_factors.items():
        rinn_name = hf_to_rinn_name(name, n_layers)
        if rinn_name:
            npz_data[f"{rinn_name}.smooth_scale"] = factors

    np.savez(out_path, **npz_data)
    print(f"Calibration data saved: {out_path} ({len(npz_data)} arrays)")
    print(f"  Layers: {len(set(k.split('.')[0]+'.'+k.split('.')[1] for k in npz_data if '.mean_abs' in k))}")
    print(f"  Channels per layer: {len([k for k in npz_data if '.mean_abs' in k and 'h.0' in k])}")
    for k in list(npz_data.keys())[:5]:
        print(f"  {k}: {npz_data[k].shape}")

def hf_to_rinn_name(hf_name, n_layers):
    """Convert HuggingFace internal module name to RINN tensor name."""
    parts = hf_name.split('.')
    # Example: "model.layers.0.self_attn.q_proj" → "transformer.h.0.attn.w_q"
    if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers':
        try:
            l = int(parts[2])
        except ValueError:
            return None
        layer_part = '.'.join(parts[3:])
        rinn_map = {
            "self_attn.q_proj": f"transformer.h.{l}.attn.w_q",
            "self_attn.k_proj": f"transformer.h.{l}.attn.w_k",
            "self_attn.v_proj": f"transformer.h.{l}.attn.w_v",
            "self_attn.o_proj": f"transformer.h.{l}.attn.w_o",
            "mlp.gate_proj":    f"transformer.h.{l}.mlp.w1",
            "mlp.up_proj":      f"transformer.h.{l}.mlp.w3",
            "mlp.down_proj":    f"transformer.h.{l}.mlp.w2",
        }
        return rinn_map.get(layer_part)
    # Global layers
    if hf_name == "model.embed_tokens":
        return "transformer.wte"
    if hf_name == "model.norm":
        return "transformer.ln_f"
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate per-channel scales for q4 quantization")
    parser.add_argument("model_dir", help="HF model directory")
    parser.add_argument("output_dir", help="Output directory for calibration data")
    parser.add_argument("--calib-data", default=None,
                        help="Path to .npy calibration data (token IDs)")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Number of calibration sequences")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for calibration")

    args = parser.parse_args()

    # Default calibration data
    calib_path = args.calib_data
    if calib_path is None:
        # Try to find calibration data in the project
        candidates = [
            "/home/aquama/Development/RINA_Project/data/dclm_pretrain_llama_2000m.npy",
            "/home/aquama/Development/RINA_Project/data/mix_pretrain_llama_v3.npy",
        ]
        for c in candidates:
            if os.path.exists(c):
                calib_path = c
                break

    if calib_path is None or not os.path.exists(calib_path):
        print("ERROR: No calibration data found. Use --calib-data to specify a .npy file.")
        sys.exit(1)

    calibrate(args.model_dir, args.output_dir, calib_path,
              n_samples=args.n_samples, seq_len=args.seq_len)
