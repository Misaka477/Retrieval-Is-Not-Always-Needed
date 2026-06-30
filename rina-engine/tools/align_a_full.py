#!/usr/bin/env python3
"""Compare engine test_a_align output (all 16 MLA layers + lm_head) vs PyTorch."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, json, sys, math

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

def apply_rope_flat(x, cos, sin, B, T, H, d):
    x_f = x.reshape(B*T*H, d)
    half = d // 2
    ct = cos[:T].reshape(T, half).repeat_interleave(H, dim=0)
    st = sin[:T].reshape(T, half).repeat_interleave(H, dim=0)
    x0, x1 = x_f[:, ::2], x_f[:, 1::2]
    return torch.stack([x0*ct-x1*st, x0*st+x1*ct], dim=-1).flatten(-2).reshape(B, T, H*d)

def main():
    cfg, tensors = load_rinn_weights("/tmp/model_a.rinn")
    d=cfg['dim']; H=cfg['n_heads']; Hkv=cfg['n_kv_heads']
    dh=cfg['head_dim']; dhr=cfg.get('d_h_r',32); dc=cfg.get('d_c',160)
    L=cfg['n_layers']; V=cfg['vocab_size']; hd=d*4*2//3//256*256
    dq=dh+dhr
    print(f"L={L} d={d} H={H} Hkv={Hkv} dh={dh} dhr={dhr} dc={dc} hd={hd}\n")

    # Build pure fp32 PyTorch model
    from collections import OrderedDict
    class L3XTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, d)
            self.ln_f = nn.LayerNorm(d, bias=False, eps=1e-5)
            self.lm_head = nn.Linear(d, V, bias=False)
            self.ln1 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.path_ln = nn.ModuleList([nn.LayerNorm(dc, bias=False, eps=1e-5) for _ in range(L)])
            self.w_dqkv = nn.ModuleList([nn.Linear(d, dc, bias=False) for _ in range(L)])
            self.w_uq = nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(L)])
            self.w_uk = nn.ModuleList([nn.Linear(dc, Hkv*dh, bias=False) for _ in range(L)])
            self.w_k2v = nn.ModuleList([nn.Linear(Hkv*dh, Hkv*dh, bias=False) for _ in range(L)])
            self.w_qr = nn.ModuleList([nn.Linear(d, H*dhr, bias=False) for _ in range(L)])
            self.w_kr = nn.ModuleList([nn.Linear(d, Hkv*dhr, bias=False) for _ in range(L)])
            self.c_proj = nn.ModuleList([nn.Linear(H*dh, d, bias=False) for _ in range(L)])
            self.mlp_w1 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
            self.mlp_w2 = nn.ModuleList([nn.Linear(hd, d, bias=False) for _ in range(L)])
            self.mlp_w3 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
            self.load_weights()
        def load_weights(self):
            # Map from .rinn naming to PyTorch naming
            for l in range(L):
                prefix = f"transformer.h.{l}."
                mapping = {
                    f"{prefix}ln1.weight": self.ln1[l].weight,
                    f"{prefix}ln2.weight": self.ln2[l].weight,
                    f"{prefix}path.q_norm.weight": self.path_ln[l].weight,
                    f"{prefix}path.w_dqkv.weight": self.w_dqkv[l].weight,
                    f"{prefix}path.w_uq.weight": self.w_uq[l].weight,
                    f"{prefix}path.w_uk.weight": self.w_uk[l].weight,
                    f"{prefix}path.w_k2v.weight": self.w_k2v[l].weight,
                    f"{prefix}path.w_qr.weight": self.w_qr[l].weight,
                    f"{prefix}path.w_kr.weight": self.w_kr[l].weight,
                    f"{prefix}path.c_proj.weight": self.c_proj[l].weight,
                    f"{prefix}mlp.w1.weight": self.mlp_w1[l].weight,
                    f"{prefix}mlp.w2.weight": self.mlp_w2[l].weight,
                    f"{prefix}mlp.w3.weight": self.mlp_w3[l].weight,
                }
                for rnn, param in mapping.items():
                    if rnn in tensors:
                        param.data.copy_(tensors[rnn].float().cuda())
            # Global weights
            self.wte.weight.data.copy_(tensors["transformer.wte.weight"].float().cuda())
            self.ln_f.weight.data.copy_(tensors["transformer.ln_f.weight"].float().cuda())
            self.lm_head.weight.data.copy_(tensors["lm_head.weight"].float().cuda())
        def forward(self, ids):
            B,T = ids.shape
            h = self.wte(ids)
            for l in range(L):
                # LN1 + cq
                a = self.ln1[l](h)
                cq = self.path_ln[l](self.w_dqkv[l](a))
                # qc, kc, v
                qc = self.w_uq[l](cq)
                kc = self.w_uk[l](cq)
                v = self.w_k2v[l](kc)
                # RoPE qr, kr
                qr = apply_rope_flat(self.w_qr[l](a),
                    tensors[f"transformer.h.{l}.path.rope_q.cos"].cuda(),
                    tensors[f"transformer.h.{l}.path.rope_q.sin"].cuda(), B,T,H,dhr)
                kr = apply_rope_flat(self.w_kr[l](a),
                    tensors[f"transformer.h.{l}.path.rope.cos"].cuda(),
                    tensors[f"transformer.h.{l}.path.rope.sin"].cuda(), B,T,Hkv,dhr)
                # Build QKV
                rep = H//Hkv
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
                # Flash attention
                scale=1.0/math.sqrt(dq)
                attn=torch.bmm(Qf*scale,Kf.transpose(1,2))
                mask=torch.triu(torch.ones(T,T,device='cuda')*float('-inf'),diagonal=1)
                O=torch.bmm(F.softmax(attn+mask.unsqueeze(0),dim=-1),Vf)
                h = h + self.c_proj[l](O.reshape(B,H,T,dh).transpose(1,2).reshape(B,T,H*dh))
                # MLP
                a2 = self.ln2[l](h)
                h = h + self.mlp_w2[l](F.silu(self.mlp_w1[l](a2))*self.mlp_w3[l](a2))
            return self.lm_head(self.ln_f(h))

    model = L3XTorch().cuda().eval()
    ids = torch.tensor([[128000, 9906, 11]]).cuda()
    with torch.no_grad():
        logits_pt = model(ids)
    logits_eng = torch.from_numpy(np.fromfile("/tmp/a_logits.bin", dtype=np.float32)).cuda().reshape(1,3,-1)
    diff = (logits_pt - logits_eng).abs()
    print(f"Logit comparison:")
    print(f"  max_diff = {diff.max():.6e}")
    print(f"  mean_diff = {diff.mean():.6e}")
    pt_tok = logits_pt[0,-1].argmax().item()
    eng_tok = logits_eng[0,-1].argmax().item()
    print(f"  PT argmax  = {pt_tok}")
    print(f"  Eng argmax = {eng_tok}")
    ok = diff.max() < 0.1
    print(f"\n{'✅ 16-layer alignment PASSED!' if ok else '❌ FAILED'} (max_diff={diff.max():.4f})")

if __name__ == '__main__':
    main()
