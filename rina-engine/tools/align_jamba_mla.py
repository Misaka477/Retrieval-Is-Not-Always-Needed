#!/usr/bin/env python3
"""Compare engine test_jamba_mla_step output (Jamba MLA layer 3) vs PyTorch."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, json, math

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
    return torch.from_numpy(np.fromfile(path, dtype=np.float32))

def compare(name, a, b):
    if torch.is_tensor(a): a = a.detach().cpu().numpy()
    if torch.is_tensor(b): b = b.detach().cpu().numpy()
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    diff = np.abs(a - b)
    md = diff.max(); me = diff.mean(); idx = np.argmax(diff)
    print(f"  {name:24s} max={md:.2e}  mean={me:.2e}  @{idx} (a={a[idx]:.4f} b={b[idx]:.4f})")
    return md

def apply_rope_flat(x, cos, sin, B, T, H, d):
    x_f = x.reshape(B*T*H, d)
    half = d // 2
    ct = cos[:T].reshape(T, half).repeat_interleave(H, dim=0)
    st = sin[:T].reshape(T, half).repeat_interleave(H, dim=0)
    x0, x1 = x_f[:, ::2], x_f[:, 1::2]
    return torch.stack([x0*ct-x1*st, x0*st+x1*ct], dim=-1).flatten(-2).reshape(B, T, H*d)

def build_qkv_from(qc, qr, kc, kr, v, B, T, H, Hkv, dh, dhr, dq):
    rep = H // Hkv
    qc_n=qc.reshape(B*T,H,dh); qr_n=qr.reshape(B*T,H,dhr)
    kc_n=kc.reshape(B*T,Hkv,dh); kr_n=kr.reshape(B*T,Hkv,dhr)
    v_n=v.reshape(B*T,Hkv,dh)
    Qf=torch.zeros(B*H,T,dq,device='cuda'); Kf=torch.zeros(B*H,T,dq,device='cuda'); Vf=torch.zeros(B*H,T,dh,device='cuda')
    for bh in range(B*H):
        b=bh//H; hh=bh%H; kv=hh//rep
        for t in range(T):
            idx=b*T+t
            Qf[bh,t,:dh]=qc_n[idx,hh,:]; Qf[bh,t,dh:]=qr_n[idx,hh,:]
            Kf[bh,t,:dh]=kc_n[idx,kv,:]; Kf[bh,t,dh:]=kr_n[idx,kv,:]
            Vf[bh,t,:]=v_n[idx,kv,:]
    return Qf, Kf, Vf

def main():
    cfg, tensors = load_rinn_weights("/tmp/jamba_qw2.rinn")
    d=cfg['dim']; H=cfg['n_heads']; Hkv=cfg['n_kv_heads']
    dh=cfg['head_dim']; dhr=cfg.get('d_h_r',32); dc=cfg.get('d_c',160)
    dq=dh+dhr; hd=d*4*2//3//256*256
    B,T=1,3; n=B*T
    print(f"d={d} H={H} Hkv={Hkv} dh={dh} dhr={dhr} dc={dc}\n")

    l=3
    w_dqkv = tensors[f"transformer.h.{l}.path.w_dqkv.weight"].cuda()
    q_norm_w = tensors[f"transformer.h.{l}.path.q_norm.weight"].cuda()
    w_uq = tensors[f"transformer.h.{l}.path.w_uq.weight"].cuda()
    w_uk = tensors[f"transformer.h.{l}.path.w_uk.weight"].cuda()
    w_k2v = tensors[f"transformer.h.{l}.path.w_k2v.weight"].cuda()
    w_qr = tensors[f"transformer.h.{l}.path.w_qr.weight"].cuda()
    w_kr = tensors[f"transformer.h.{l}.path.w_kr.weight"].cuda()
    c_proj = tensors[f"transformer.h.{l}.path.c_proj.weight"].cuda()
    ln1_w = tensors[f"transformer.h.{l}.ln1.weight"].cuda()
    ln2_w = tensors[f"transformer.h.{l}.ln2.weight"].cuda()
    w1 = tensors[f"transformer.h.{l}.mlp.w1.weight"].cuda()
    w2 = tensors[f"transformer.h.{l}.mlp.w2.weight"].cuda()
    w3 = tensors[f"transformer.h.{l}.mlp.w3.weight"].cuda()
    cos_q = tensors[f"transformer.h.{l}.path.rope_q.cos"].float().cuda()
    sin_q = tensors[f"transformer.h.{l}.path.rope_q.sin"].float().cuda()
    cos_k = tensors[f"transformer.h.{l}.path.rope.cos"].float().cuda()
    sin_k = tensors[f"transformer.h.{l}.path.rope.sin"].float().cuda()

    # Engine dumps
    eng_h_in = load_bin("/tmp/jm_l3_h_in").reshape(B,T,d)
    eng_a_ln1 = load_bin("/tmp/jm_l3_a_ln1").reshape(B,T,d)
    eng_cq = load_bin("/tmp/jm_l3_cq").reshape(B,T,dc)
    eng_qc = load_bin("/tmp/jm_l3_qc").reshape(B,T,H*dh)
    eng_kc = load_bin("/tmp/jm_l3_kc").reshape(B,T,Hkv*dh)
    eng_v = load_bin("/tmp/jm_l3_v").reshape(B,T,Hkv*dh)
    eng_qr = load_bin("/tmp/jm_l3_qr").reshape(B,T,H*dhr)
    eng_kr = load_bin("/tmp/jm_l3_kr").reshape(B,T,Hkv*dhr)
    eng_qr_rope = load_bin("/tmp/jm_l3_qr_rope").reshape(B,T,H*dhr)
    eng_kr_rope = load_bin("/tmp/jm_l3_kr_rope").reshape(B,T,Hkv*dhr)
    eng_Qf = load_bin("/tmp/jm_l3_Qf").reshape(B*H,T,dq)
    eng_Kf = load_bin("/tmp/jm_l3_Kf").reshape(B*H,T,dq)
    eng_Vf = load_bin("/tmp/jm_l3_Vf").reshape(B*H,T,dh)
    eng_h_aft = load_bin("/tmp/jm_l3_h_aft").reshape(B,T,d)
    eng_h_final = load_bin("/tmp/jm_l3_h_final").reshape(B,T,d)

    print("═══ PyTorch reference (Jamba MLA layer 3) ═══\n")

    h = eng_h_in.cuda()  # use engine's h (from layers 0-2 forward)
    ln1 = nn.LayerNorm(d,bias=False,eps=1e-5).cuda(); ln1.weight.data.copy_(ln1_w)
    a = ln1(h)
    compare("[1] LN1", a, eng_a_ln1)

    cq = F.linear(a, w_dqkv)
    qn = nn.LayerNorm(dc,bias=False,eps=1e-5).cuda(); qn.weight.data.copy_(q_norm_w)
    cq = qn(cq)
    compare("[2] cq", cq, eng_cq)

    qc = F.linear(cq, w_uq); kc = F.linear(cq, w_uk); vt = F.linear(kc, w_k2v)
    compare("[3a] qc", qc, eng_qc); compare("[3b] kc", kc, eng_kc); compare("[3c] v", vt, eng_v)

    qr = F.linear(a, w_qr); kr = F.linear(a, w_kr)
    compare("[4a] qr", qr, eng_qr); compare("[4b] kr", kr, eng_kr)

    qr_r = apply_rope_flat(qr, cos_q, sin_q, B, T, H, dhr)
    kr_r = apply_rope_flat(kr, cos_k, sin_k, B, T, Hkv, dhr)
    compare("[5a] qr_rope", qr_r, eng_qr_rope); compare("[5b] kr_rope", kr_r, eng_kr_rope)

    Qf, Kf, Vf = build_qkv_from(qc, qr_r, kc, kr_r, vt, B, T, H, Hkv, dh, dhr, dq)
    compare("[6a] Qf", Qf, eng_Qf); compare("[6b] Kf", Kf, eng_Kf); compare("[6c] Vf", Vf, eng_Vf)

    scale=1.0/math.sqrt(dq)
    attn=torch.bmm(Qf*scale, Kf.transpose(1,2))
    mask=torch.triu(torch.ones(T,T,device='cuda')*float('-inf'),diagonal=1)
    O=torch.bmm(F.softmax(attn+mask.unsqueeze(0),dim=-1),Vf)
    O_r = O.reshape(B,H,T,dh).transpose(1,2).reshape(B,T,H*dh)
    cproj = F.linear(O_r, c_proj)
    h = h + cproj
    compare("[7] h_aft_attn", h, eng_h_aft)

    ln2 = nn.LayerNorm(d,bias=False,eps=1e-5).cuda(); ln2.weight.data.copy_(ln2_w)
    a2 = ln2(h)
    mlp = F.silu(F.linear(a2,w1))*F.linear(a2,w3)
    h = h + F.linear(mlp, w2)
    md = compare("[8] h_final", h, eng_h_final)
    print(f"\n{'✅ MATCH' if md < 0.1 else '❌ MISMATCH'} (max_diff={md:.4f})")

if __name__ == '__main__':
    main()
