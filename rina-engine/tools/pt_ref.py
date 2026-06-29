#!/usr/bin/env python3
"""PyTorch reference: runs the full forward using PyTorch, with optional module replacement.
Usage: python3 pt_ref.py [--custom-embed] [--custom-ln] [--custom-attn] [--custom-mlp]
"""
import torch, numpy as np, struct, argparse, subprocess
from rina.model_a import RINA_A, RINA_A_Config

def load_rinn_tensor(path, name):
    with open(path, 'rb') as f:
        hdr = f.read(512)
        idx_off, idx_sz = struct.unpack('<QQ', hdr[32:48])
        f.seek(idx_off); idx = f.read(idx_sz); pos = 0
        while pos < idx_sz:
            nl = struct.unpack('<H', idx[pos:pos+2])[0]; pos += 2
            n = idx[pos:pos+nl].decode(); pos += nl
            shape = struct.unpack('<iiii', idx[pos:pos+16]); pos += 16
            qtype = idx[pos]; pos += 1; blk = struct.unpack('<H', idx[pos:pos+2])[0]; pos += 2
            offset, size = struct.unpack('<QQ', idx[pos:pos+16]); pos += 16
            if n == name:
                f.seek(offset)
                arr = np.frombuffer(f.read(size), dtype=np.float32)
                s = [x for x in shape[:2] if x > 0]
                return torch.from_numpy(arr.reshape(s).copy())
    return None

def pt_layernorm(x, w):
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, keepdim=True, unbiased=False)
    return (x - m) / (v + 1e-5).sqrt() * w

def run_pt_reference(rinn_path, ids, steps, flags):
    """Run full PyTorch reference forward. Returns list of generated tokens."""
    # Build config
    cfg = RINA_A_Config(vocab_size=128256, block_size=512, use_int4=False,
                        n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160)
    m = RINA_A(cfg).eval().cuda()
    cd = torch.load('/home/aquama/Development/RINA_Project/models/out-0.1b-a-v2/a_final.pt',
                    map_location='cpu', weights_only=False)
    sd = cd['model'] if 'model' in cd else cd
    m.load_state_dict(sd, strict=False)

    # Run with PyTorch SDPA as reference
    x = torch.tensor([ids]).cuda()
    results = []
    with torch.no_grad():
        for step in range(steps):
            l, _, _ = m(x, None)
            tok = l[0, -1].argmax().item()
            results.append(tok)
            x = torch.cat([x, torch.tensor([[tok]]).cuda()], dim=1)
    return results

def run_engine(rinn_path, ids, steps):
    """Run engine and return output"""
    proc = subprocess.run(['/home/aquama/Development/RINA_Project/rina-engine/build/rina_infer',
        '--model', rinn_path, '--ids', ' '.join(str(i) for i in ids), '--steps', str(steps)],
        capture_output=True, text=True)
    return [int(t) for t in proc.stdout.strip().split()]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rinn', default='/tmp/model_a_v2.rinn')
    ap.add_argument('--ids', default='128000 791 6864 315 9822 374')
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--custom-embed', action='store_true')
    ap.add_argument('--custom-ln', action='store_true')
    ap.add_argument('--custom-attn', action='store_true')
    ap.add_argument('--custom-mlp', action='store_true')
    ap.add_argument('--custom-cproj', action='store_true')
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split()]
    pt_ref = run_pt_reference(args.rinn, ids, args.steps, vars(args))
    eng_out = run_engine(args.rinn, ids, args.steps)
    
    print(f'PyTorch ref: {pt_ref}')
    print(f'Engine:      {eng_out}')
    ok = pt_ref == eng_out
    status = "OK" if ok else "FAIL"
    print(f'Match: {status}')
