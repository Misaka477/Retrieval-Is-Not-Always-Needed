#!/usr/bin/env python3
import struct, json, torch, numpy as np

def q4_0_block(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs).float()
    s = xf.abs().max(-1, keepdim=True).values / 7.0
    xq = (xf / (s + 1e-8)).round().clamp(-7, 7)
    return (xq * s).reshape(o)

_NON_QAT_SUFFIXES = ('q_norm.weight', 'ln1.weight', 'ln2.weight', 'ln_f.weight',
    'rope_q.cos', 'rope_q.sin', 'rope.cos', 'rope.sin')

def is_qat_weight(name):
    if name == 'transformer.wte.weight':
        return False
    if name == 'lm_head.weight':
        return True
    if any(name.endswith(s) for s in _NON_QAT_SUFFIXES):
        return False
    if name.endswith('.weight'):
        parts = name.split('.')
        if len(parts) >= 4:
            third = parts[3]
            if third in ('mlp', 'path'):
                return True
    return False

def gen_jambaqw2_config():
    return {
        "name": "rina-jamba-qw2", "dim": 640, "n_layers": 16,
        "n_heads": 10, "n_kv_heads": 5, "head_dim": 64, "d_c": 160,
        "d_h_r": 32, "vocab_size": 128256, "block_size": 512,
        "ssm_steps": 3, "quant_mode": "", "ssm_qbits": 0,
        "layers": [
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "sparse_gather_fa"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "sparse_gather_fa"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "sparse_gather_fa"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "inertia_wave_ssm"},
            {"type": "sparse_gather_fa"},
        ]
    }

def convert(ckpt_path, output_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = sd['model'] if 'model' in sd else sd
    config = gen_jambaqw2_config()
    config_json = json.dumps(config, indent=2).encode('utf-8')

    tensors = []
    data_list = []
    for name, param in state.items():
        if is_qat_weight(name):
            arr = q4_0_block(param, 32).numpy().astype(np.float32)
        else:
            arr = param.numpy().astype(np.float32)

        if name == 'transformer.wte.weight':
            print(f'  wte: raw fp32, shape={list(arr.shape)}')
        elif name == 'lm_head.weight':
            print(f'  lm_head: q4_0_block applied separately, shape={list(arr.shape)}')
        elif is_qat_weight(name):
            print(f'  {name}: q4_0_block applied')
        else:
            print(f'  {name}: raw fp32')

        n_elems = arr.size
        packed = arr.tobytes()
        shape = list(arr.shape)
        while len(shape) < 4: shape.append(0)
        tensors.append({'name': name, 'shape': shape, 'size': len(packed)})
        data_list.append(packed)

    idx_entry_size = lambda n: 2 + len(n.encode('utf-8')) + 4*4 + 1 + 2 + 8 + 8
    index_size = sum(idx_entry_size(t['name']) for t in tensors)
    data_start = ((512 + len(config_json) + index_size + 255) // 256) * 256

    offset = data_start
    for t in tensors: t['offset'] = offset; offset += t['size']

    with open(output_path, 'wb') as f:
        hdr = bytearray(512)
        hdr[0:4] = b'RINN'
        struct.pack_into('<I', hdr, 4, 1)
        struct.pack_into('<I', hdr, 8, len(tensors))
        struct.pack_into('<Q', hdr, 16, 512)
        struct.pack_into('<Q', hdr, 24, len(config_json))
        struct.pack_into('<Q', hdr, 32, 512 + len(config_json))
        struct.pack_into('<Q', hdr, 40, index_size)
        struct.pack_into('<Q', hdr, 48, data_start)
        f.write(hdr)
        f.write(config_json)

        for t in tensors:
            nb = t['name'].encode('utf-8')
            f.write(struct.pack('<H', len(nb)))
            f.write(nb)
            for d in range(4): f.write(struct.pack('<i', t['shape'][d]))
            f.write(struct.pack('<B', 6))
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<Q', t['offset']))
            f.write(struct.pack('<Q', t['size']))

        pad = data_start - f.tell()
        if pad > 0: f.write(b'\x00' * pad)

        for d in data_list: f.write(d)

    total_mb = offset / 1024 / 1024
    print(f'Written: {output_path} ({total_mb:.1f} MB, {len(tensors)} tensors)')

if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'models/out-rina-jamba-qw2/jambaqw2_final.pt'
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/model_qat.rinn'
    convert(ckpt, out)
