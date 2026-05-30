#!/usr/bin/env python3
"""
Transfer RWKV-7 FFN weights into a MoHERWKV attractor initialization checkpoint.
Usage: python weight_transfer.py

Steps:
  1. Download RWKV-7 World 0.1B from HuggingFace (if not cached).
  2. Load raw state dict and infer architecture dimensions.
  3. Create a MoHERWKV model (rina.architectures.mohe_rwkv) with matching dm.
  4. For each attractor expert, initialize proj_up / proj_down from the
     corresponding RWKV block FFN (key / value) weights.
  5. Save the resulting checkpoint.
"""
import os, sys, json, math, warnings
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rina import MoHERWKV

MODEL_ID = "BlinkDL/rwkv7-g1"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "checkpoints", "mohe_transferred_init.pt")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "hf_cache")

os.makedirs(os.path.dirname(DEST), exist_ok=True)

def download_model_weights():
    """Download RWKV .pth from HuggingFace hub if not already cached."""
    weight_filename = "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    local_path = os.path.join(CACHE_DIR, weight_filename)
    if os.path.exists(local_path):
        print(f"Found cached weights: {local_path}")
        return local_path

    print(f"Downloading {MODEL_ID} from HuggingFace ...")
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=weight_filename,
            cache_dir=CACHE_DIR,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
        )
    except ImportError:
        print("huggingface_hub not installed. "
              "Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print(f"Downloaded to: {local_path}")
    return local_path


def infer_dimensions(state):
    """Infer dm, n_layer, n_ffn, vocab from the state dict keys / shapes."""
    # --- embedding dim --------------------------------------------------
    emb_w = state.get("emb.weight")
    if emb_w is not None:
        vocab, dm = emb_w.shape
    else:
        # try token embedding
        for k in state:
            if "emb" in k and ".weight" in k:
                vocab, dm = state[k].shape
                break
        else:
            raise KeyError("Cannot locate embedding weight in state dict")
    print(f"  vocab={vocab}, dm={dm}")

    # --- number of layers -----------------------------------------------
    n_layer = 0
    for k in state:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            try:
                n_layer = max(n_layer, int(parts[1]) + 1)
            except ValueError:
                pass
    print(f"  n_layer (detected) = {n_layer}")

    # --- FFN hidden size ------------------------------------------------
    hidden_sz = None
    ffn_key_patterns = [".ffn.key.weight", ".ffn.key.weight",  # common RWKV-7
                        ".channel_mix.key.weight",
                        ".ffn.xx.weight"]  # fallback
    for k, v in state.items():
        if any(p in k for p in [".ffn.key.weight", ".channel_mix.key.weight"]):
            hidden_sz, _ = v.shape
            break
    if hidden_sz is None:
        # fallback: guess 4*dm
        hidden_sz = 4 * dm
        print(f"  n_ffn not found, assuming 4*dm = {hidden_sz}")
    else:
        print(f"  n_ffn (hidden_sz) = {hidden_sz}")

    return dm, n_layer, hidden_sz, vocab


def find_ffn_layers(state, n_layer):
    """Return list of (key_weight, value_weight) per block, best-effort."""
    ffn_pairs = []
    for i in range(n_layer):
        kw = None
        vw = None
        # Search common naming patterns
        for prefix in [f"blocks.{i}.ffn", f"blocks.{i}.channel_mix"]:
            kk = f"{prefix}.key.weight"
            vk = f"{prefix}.value.weight"
            if kk in state and vk in state:
                kw = state[kk]
                vw = state[vk]
                break
            # also try .weight directly on channel_mix
        if kw is None:
            # fallback: try any weight-containing key in blocks.{i}
            for k, v in state.items():
                if k.startswith(f"blocks.{i}") and "key" in k and ".weight" in k:
                    kw = v
                if k.startswith(f"blocks.{i}") and "value" in k and ".weight" in k:
                    vw = v
        if kw is not None and vw is not None:
            ffn_pairs.append((kw, vw))
        else:
            print(f"  Warning: block {i} FFN not found, skipping")
    return ffn_pairs


def init_attractor_from_ffn(attractor, key_w, value_w, dm):
    """Copy RWKV FFN key/value weights into attractor proj_up / proj_down."""

    # proj[0] = Linear(dm, dm*2)  -> key_w = (hidden_sz, dm)
    # Weight shapes:
    #   proj[0].weight = (dm*2, dm)   proj_up
    #   proj[2].weight = (dm, dm*2)   proj_down

    hidden_sz, _ = key_w.shape
    target_hidden = dm * 2

    proj_up = attractor.proj[0].weight  # shape (dm*2, dm)
    proj_down = attractor.proj[2].weight  # shape (dm, dm*2)

    if hidden_sz == target_hidden:
        proj_up.data.copy_(key_w)
        proj_down.data.copy_(value_w)
        print(f"    Full direct copy: hidden_sz={hidden_sz} == dm*2")
    elif hidden_sz > target_hidden:
        # Truncate excess hidden dimensions
        proj_up.data.copy_(key_w[:target_hidden, :])
        proj_down.data.copy_(value_w[:, :target_hidden])
        print(f"    Truncated: hidden_sz={hidden_sz} -> dm*2={target_hidden}")
    else:
        # Pad with small noise
        with torch.no_grad():
            proj_up.data[:hidden_sz, :] = key_w
            proj_up.data[hidden_sz:, :] = torch.randn(target_hidden - hidden_sz, dm) * 0.02
            proj_down.data[:, :hidden_sz] = value_w
            proj_down.data[:, hidden_sz:] = torch.randn(dm, target_hidden - hidden_sz) * 0.02
        print(f"    Padded: hidden_sz={hidden_sz} < dm*2={target_hidden}")

    return attractor


def main():
    print("=" * 60)
    print("RWKV-7 → MoHERWKV Weight Transfer")
    print("=" * 60)

    # 1. Download
    weights_path = download_model_weights()

    # 2. Load
    print("\nLoading state dict ...")
    # RWKV .pth is a plain dict (not wrapped in model class), load directly
    try:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(weights_path, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        state = raw.state_dict() if hasattr(raw, "state_dict") else raw
    print(f"  Total keys: {len(state)}")

    # 3. Infer dimensions
    print("\nInferring architecture ...")
    dm, n_layer, hidden_sz, vocab = infer_dimensions(state)

    # 4. Locate FFN layer pairs
    print("\nExtracting FFN layer weights ...")
    ffn_pairs = find_ffn_layers(state, n_layer)
    n_experts = len(ffn_pairs)
    print(f"  Found FFN pairs: {n_experts} blocks")

    if n_experts == 0:
        print("ERROR: No FFN weights found in state dict.")
        print("Known keys (sample):")
        for k in sorted(state.keys())[:20]:
            print(f"  {k}")
        sys.exit(1)

    # 5. Create MoHERWKV with matching dm
    #    np (pattern count) is set proportionally; 512 is a good default
    np_ = dm * 2
    print(f"\nCreating MoHERWKV(vocab={vocab}, dm={dm}, np={np_}, "
          f"n_experts={n_experts}) ...")
    model = MoHERWKV(vocab, dm, np_, n_experts=n_experts,
                     aux_loss_weight=0.5, route_noise=0.2, topk=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MoHERWKV total params: {total_params / 1e6:.2f}M")

    # 6. Initialize each attractor expert from corresponding RWKV block FFN
    print("\nInitializing attractor experts from RWKV FFN weights ...")
    transferred = 0
    for i, (key_w, value_w) in enumerate(ffn_pairs):
        if i >= len(model.experts):
            print(f"  Warning: more blocks than experts, stopping at expert {i}")
            break
        expert = model.experts[i]
        init_attractor_from_ffn(expert, key_w, value_w, dm)
        transferred += key_w.numel() + value_w.numel()
        print(f"  Expert {i}: transferred {key_w.numel() + value_w.numel()} params")

    print(f"\nTotal parameters transferred: {transferred:,} "
          f"({transferred / total_params * 100:.1f}% of MoHE model)")

    # 7. Save
    print(f"\nSaving checkpoint to {DEST} ...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab": vocab,
            "dm": dm,
            "np": np_,
            "n_experts": n_experts,
            "source_model": MODEL_ID,
            "transferred_params": transferred,
        },
        "rwkv_ffn_pairs": n_experts,
        "total_model_params_m": total_params / 1e6,
    }, DEST)
    print(f"  Saved ({os.path.getsize(DEST) / 1024 / 1024:.1f} MB)")

    # 8. Summary
    print("\n" + "=" * 60)
    print("Transfer complete.")
    print(f"  RWKV source  : {MODEL_ID}")
    print(f"  MoHE dm      : {dm}, np={np_}, experts={n_experts}")
    print(f"  FFN hidden_sz: {hidden_sz}")
    print(f"  Transferred  : {transferred:,} parameters")
    print(f"  MoHE total   : {total_params:,} parameters")
    print(f"  Checkpoint   : {DEST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
