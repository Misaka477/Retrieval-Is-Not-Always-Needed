#!/usr/bin/env python3
"""Compare engine test_jamba_step output vs PyTorch reference (SSM layer 0)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, json, sys, math, os

def load_rinn_weights(path):
    with open(path, 'rb') as f:
        hdr = f.read(512)
        n_tensors = struct.unpack('<I', hdr[8:12])[0]
        idx_off = struct.unpack('<Q', hdr[32:40])[0]
        idx_sz = struct.unpack('<Q', hdr[40:48])[0]
        f.seek(512)
        cfg = json.loads(f.read(struct.unpack('<Q', hdr[24:32])[0]))
        f.seek(idx_off); idx = f.read(idx_sz)
        pos = 0; tensors = {}
        while pos < idx_sz:
            nl = struct.unpack('<H', idx[pos:pos+2])[0]; pos += 2
            name = idx[pos:pos+nl].decode(); pos += nl
            shape = struct.unpack('<iiii', idx[pos:pos+16]); pos += 16
            qtype = idx[pos]; pos += 3
            offset, sz = struct.unpack('<QQ', idx[pos:pos+16]); pos += 16
            f.seek(offset)
            arr = np.frombuffer(f.read(sz), dtype=np.float32)
            tensors[name] = torch.from_numpy(arr.reshape([x for x in shape if x > 0]))
        return cfg, tensors

def load_bin(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32)).float()

def compare(name, a, b):
    if torch.is_tensor(a): a = a.detach().cpu().numpy()
    if torch.is_tensor(b): b = b.detach().cpu().numpy()
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    diff = np.abs(a - b)
    md = diff.max(); me = diff.mean(); idx = np.argmax(diff)
    print(f"  {name:24s} max={md:.2e}  mean={me:.2e}  @{idx} (a={a[idx]:.4f} b={b[idx]:.4f})")
    return md

def main():
    ids = [128000, 9906, 11]
    B, T = 1, 3; n = B * T

    cfg, tensors = load_rinn_weights("/tmp/jamba_qw2.rinn")
    d = cfg['dim']; H = cfg['n_heads']; dh = cfg['head_dim']
    dc = cfg['d_c']; ss = cfg.get('ssm_steps', 3); hd = d*4*2//3//256*256
    print(f"d={d} H={H} dh={dh} dc={dc} ss={ss} hd={hd}")

    wte = tensors["transformer.wte.weight"].cuda()
    w_dq = tensors["transformer.h.0.path.w_dq.weight"].cuda()
    q_norm_w = tensors["transformer.h.0.path.q_norm.weight"].cuda()
    w_mem = [tensors[f"transformer.h.0.path.w_mem.{k}.weight"].cuda() for k in range(ss)]
    w_decay = [tensors[f"transformer.h.0.path.w_decay.{k}.weight"].cuda() for k in range(ss)]
    w_out = tensors["transformer.h.0.path.w_out.weight"].cuda()
    ln1_w = tensors["transformer.h.0.ln1.weight"].cuda()
    ln2_w = tensors["transformer.h.0.ln2.weight"].cuda()
    w1 = tensors["transformer.h.0.mlp.w1.weight"].cuda()
    w2 = tensors["transformer.h.0.mlp.w2.weight"].cuda()
    w3 = tensors["transformer.h.0.mlp.w3.weight"].cuda()

    # Engine dumps
    eng_embed = load_bin("/tmp/j_embed").reshape(B,T,d)
    eng_a_ln1 = load_bin("/tmp/j_l0_a_ln1").reshape(B,T,d)
    eng_a_before_concat = load_bin("/tmp/j_l0_a_before_concat").reshape(B,T,d)
    compare("[0] a before concat vs a_ln1", eng_a_before_concat, eng_a_ln1)
    eng_m_after_copy1 = load_bin("/tmp/j_l0_m_after_copy1").reshape(B,T,d)
    compare("[0] m after copy1 vs a_ln1", eng_m_after_copy1, eng_a_ln1)
    eng_cq = load_bin("/tmp/j_l0_cq").reshape(B,T,dc)
    eng_mems = load_bin("/tmp/j_l0_mems").reshape(ss,B,T,H,dh)
    eng_db = load_bin("/tmp/j_l0_db").reshape(ss,B,T,H)
    eng_da = load_bin("/tmp/j_l0_da").reshape(3,B,T,H)
    eng_ma = load_bin("/tmp/j_l0_ma").reshape(B,T,H,dh)
    eng_sf = load_bin("/tmp/j_l0_sf").reshape(B,T,H,dh)
    eng_concat = load_bin("/tmp/j_l0_concat").reshape(B,T,d+H*dh)
    eng_wout_out = load_bin("/tmp/j_l0_wout_out").reshape(B,T,d)
    eng_h_aft_attn = load_bin("/tmp/j_l0_h_aft_attn").reshape(B,T,d)
    eng_h_final = load_bin("/tmp/j_l0_h_final").reshape(B,T,d) if os.path.exists("/tmp/j_l0_h_final") else load_bin("/tmp/j_l0_h_out").reshape(B,T,d)
    eng_mlp_out = load_bin("/tmp/j_l0_mlp_out").reshape(B,T,d)

    print("\n═══ PyTorch reference (SSM layer 0) ═══\n")

    # 1. Embedding
    h = F.embedding(torch.tensor(ids).cuda(), wte).unsqueeze(0)
    compare("[1] embed", h, eng_embed)

    # 2. LN1
    ln1 = nn.LayerNorm(d, bias=False, eps=1e-5).cuda()
    ln1.weight.data.copy_(ln1_w)
    a = ln1(h)
    compare("[2] ln1_out (a)", a, eng_a_ln1)

    # 3. w_dq + q_norm → cq
    cq = F.linear(a, w_dq)
    qn = nn.LayerNorm(dc, bias=False, eps=1e-5).cuda()
    qn.weight.data.copy_(q_norm_w)
    cq = qn(cq)
    compare("[3] cq", cq, eng_cq)

    # 4. SSM loop
    mems = []; decays_raw = []
    for k in range(ss):
        mems.append(F.linear(cq, w_mem[k]).view(B,T,H,dh))
        decays_raw.append(F.linear(cq, w_decay[k]))
    for k in range(ss):
        compare(f"[4a] db[{k}]", decays_raw[k], eng_db[k])
    decays = [torch.sigmoid(d) for d in decays_raw]
    for k in range(ss):
        compare(f"[4b] mems[{k}]", mems[k], eng_mems[k])

    # 5. ssm_agg: da = d0*d1*d2, ma = m0*d1*d2 + m1*d2 + m2
    da = decays[0] * decays[1] * decays[2]
    d1e = decays[1].unsqueeze(-1); d2e = decays[2].unsqueeze(-1)
    ma = mems[0]*d1e*d2e + mems[1]*d2e + mems[2]
    compare("[5a] da", da, eng_da[0])
    compare("[5b] ma", ma, eng_ma)

    # 6. SSM scan — element-by-element matching engine kernel
    ma_2d = ma.reshape(B, T, H*dh)  # [1,T,H*dh]
    da_2d = da.reshape(B, T, H)     # [1,T,H]
    sf = torch.zeros_like(ma_2d)
    for bh in range(B * H):
        bb = bh // H
        hh = bh % H
        for dd in range(dh):
            lcs = 0.0
            wcs = 0.0
            for t in range(T):
                dv = da_2d[bb, t, hh].item()
                if dv <= 1e-38: dv = 1e-38
                lcs += math.log(dv)
                ca = math.exp(lcs)
                wcs += ma_2d[bb, t, hh*dh + dd].item() / max(ca, 1e-30)
                sf[bb, t, hh*dh + dd] = ca * wcs
    compare("[6] sf (scan)", sf, eng_sf.reshape(B,T,H*dh))

    # 7. w_out projection: concat([a, sf]) → w_out
    # TRICK: compare concat built from ENGINE's a+sf vs eng_concat directly
    concat_from_eng = torch.cat([eng_a_ln1, eng_sf.reshape(B,T,H*dh)], dim=-1)
    compare("[7a] concat(eng_a+sf) vs eng", concat_from_eng, eng_concat)
    # Now compare MY a+sf vs eng_concat
    concat = torch.cat([a, sf.reshape(B,T,H*dh)], dim=-1)
    compare("[7a] concat(my_a+sf) vs eng", concat, eng_concat)
    wout_pt = F.linear(concat, w_out)
    compare("[7b] w_out output", wout_pt, eng_wout_out)
    h_aft = h + wout_pt
    compare("[7c] h after attn", h_aft, eng_h_aft_attn)

    # 8. LN2 + MLP
    ln2 = nn.LayerNorm(d, bias=False, eps=1e-5).cuda()
    ln2.weight.data.copy_(ln2_w)
    a2 = ln2(h_aft)
    gate = F.linear(a2, w1)
    up = F.linear(a2, w3)
    mlp_h = F.silu(gate) * up
    mlp_out = F.linear(mlp_h, w2)
    compare("[8a] mlp_out", mlp_out, eng_mlp_out)
    h_final = h_aft + mlp_out
    compare("[8b] h_final", h_final, eng_h_final)

    print(f"\n═══ FINAL ═══")
    md = compare("h_final vs engine", h_final, eng_h_final)
    print(f"\n{'✅ MATCH' if md < 0.1 else '❌ MISMATCH'} (max_diff={md:.4f})")

if __name__ == '__main__':
    main()
