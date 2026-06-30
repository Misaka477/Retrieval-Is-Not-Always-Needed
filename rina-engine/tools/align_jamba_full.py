#!/usr/bin/env python3
"""Compare engine test_jamba_align output vs PyTorch for full Jamba model."""
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

def build_qkv(qc, qr, kc, kr, v, B, T, H, Hkv, dh, dhr, dq):
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

class SSMScan(nn.Module):
    def forward(self, mem, decay):
        B,T,Hdh=mem.shape; H=decay.shape[-1]; dh=Hdh//H
        dv=decay.clamp(min=1e-38)
        lcs=torch.cumsum(dv.log(),dim=1); ca=lcs.exp().clamp(min=1e-30)
        ca_exp=ca.unsqueeze(-1).expand(-1,-1,-1,dh).reshape(B,T,Hdh)
        return ca_exp*torch.cumsum(mem/ca_exp,dim=1)

def main():
    cfg, tensors = load_rinn_weights("/tmp/jamba_qw2.rinn")
    d=cfg['dim']; H=cfg['n_heads']; Hkv=cfg['n_kv_heads']
    dh=cfg['head_dim']; dhr=cfg.get('d_h_r',32); dc=cfg.get('d_c',160); ss=cfg.get('ssm_steps',3)
    L=cfg['n_layers']; V=cfg['vocab_size']; hd=d*4*2//3//256*256; dq=dh+dhr
    print(f"L={L} d={d} H={H} Hkv={Hkv} hd={hd}")

    class Jamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, d)
            self.ln_f = nn.LayerNorm(d, bias=False, eps=1e-5)
            self.lm_head = nn.Linear(d, V, bias=False)
            self.ln1 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.mlp_w1 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
            self.mlp_w2 = nn.ModuleList([nn.Linear(hd, d, bias=False) for _ in range(L)])
            self.mlp_w3 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
            self.path_ln = nn.ModuleList([nn.LayerNorm(dc, bias=False, eps=1e-5) for _ in range(L)])
            # SSM weights
            self.w_dq = nn.ModuleList([nn.Linear(d, dc, bias=False) for _ in range(L)])
            self.w_mem = nn.ModuleList([nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(ss)]) for _ in range(L)])
            self.w_decay = nn.ModuleList([nn.ModuleList([nn.Linear(dc, H, bias=False) for _ in range(ss)]) for _ in range(L)])
            self.w_out = nn.ModuleList([nn.Linear(d+H*dh, d, bias=False) for _ in range(L)])
            # MLA weights
            self.w_dqkv = nn.ModuleList([nn.Linear(d, dc, bias=False) for _ in range(L)])
            self.w_uq = nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(L)])
            self.w_uk = nn.ModuleList([nn.Linear(dc, Hkv*dh, bias=False) for _ in range(L)])
            self.w_k2v = nn.ModuleList([nn.Linear(Hkv*dh, Hkv*dh, bias=False) for _ in range(L)])
            self.w_qr = nn.ModuleList([nn.Linear(d, H*dhr, bias=False) for _ in range(L)])
            self.w_kr = nn.ModuleList([nn.Linear(d, Hkv*dhr, bias=False) for _ in range(L)])
            self.c_proj = nn.ModuleList([nn.Linear(H*dh, d, bias=False) for _ in range(L)])
            self.load()
        def load(self):
            for l in range(L):
                p = f"transformer.h.{l}."
                # Determine layer type from config
                is_ssm = 'ssm' in cfg['layers'][l]['type']
                if is_ssm:
                    for k in range(ss):
                        self.w_mem[l][k].weight.data.copy_(tensors[f"{p}path.w_mem.{k}.weight"].cuda().float())
                        self.w_decay[l][k].weight.data.copy_(tensors[f"{p}path.w_decay.{k}.weight"].cuda().float())
                    self.w_dq[l].weight.data.copy_(tensors[f"{p}path.w_dq.weight"].cuda().float())
                    self.w_out[l].weight.data.copy_(tensors[f"{p}path.w_out.weight"].cuda().float())
                else:
                    self.w_dqkv[l].weight.data.copy_(tensors[f"{p}path.w_dqkv.weight"].cuda().float())
                    self.w_uq[l].weight.data.copy_(tensors[f"{p}path.w_uq.weight"].cuda().float())
                    self.w_uk[l].weight.data.copy_(tensors[f"{p}path.w_uk.weight"].cuda().float())
                    self.w_k2v[l].weight.data.copy_(tensors[f"{p}path.w_k2v.weight"].cuda().float())
                    self.w_qr[l].weight.data.copy_(tensors[f"{p}path.w_qr.weight"].cuda().float())
                    self.w_kr[l].weight.data.copy_(tensors[f"{p}path.w_kr.weight"].cuda().float())
                    self.c_proj[l].weight.data.copy_(tensors[f"{p}path.c_proj.weight"].cuda().float())
                self.path_ln[l].weight.data.copy_(tensors[f"{p}path.q_norm.weight"].cuda().float())
                self.ln1[l].weight.data.copy_(tensors[f"{p}ln1.weight"].cuda().float())
                self.ln2[l].weight.data.copy_(tensors[f"{p}ln2.weight"].cuda().float())
                self.mlp_w1[l].weight.data.copy_(tensors[f"{p}mlp.w1.weight"].cuda().float())
                self.mlp_w2[l].weight.data.copy_(tensors[f"{p}mlp.w2.weight"].cuda().float())
                self.mlp_w3[l].weight.data.copy_(tensors[f"{p}mlp.w3.weight"].cuda().float())
            self.wte.weight.data.copy_(tensors["transformer.wte.weight"].cuda().float())
            self.ln_f.weight.data.copy_(tensors["transformer.ln_f.weight"].cuda().float())
            self.lm_head.weight.data.copy_(tensors["lm_head.weight"].cuda().float())
        def forward(self, ids):
            B,T = ids.shape
            h = self.wte(ids)
            for l in range(L):
                is_ssm = 'ssm' in cfg['layers'][l]['type']
                a = self.ln1[l](h)
                cq = self.path_ln[l](self.w_dq[l](a) if is_ssm else self.w_dqkv[l](a))
                if is_ssm:
                    mems = [self.w_mem[l][k](cq).view(B,T,H,dh) for k in range(ss)]
                    dcs = [torch.sigmoid(self.w_decay[l][k](cq)) for k in range(ss)]
                    d1e=dcs[1].unsqueeze(-1); d2e=dcs[2].unsqueeze(-1)
                    ma = mems[0]*d1e*d2e + mems[1]*d2e + mems[2]
                    da = dcs[0]*dcs[1]*dcs[2]
                    sf = SSMScan()(ma.reshape(B,T,H*dh), da)
                    concat = torch.cat([a, sf], dim=-1)
                    h = h + self.w_out[l](concat)
                else:
                    qc = self.w_uq[l](cq)
                    kc = self.w_uk[l](cq)
                    v = self.w_k2v[l](kc)
                    qr = apply_rope_flat(self.w_qr[l](a),
                        tensors[f"transformer.h.{l}.path.rope_q.cos"].cuda(),
                        tensors[f"transformer.h.{l}.path.rope_q.sin"].cuda(), B,T,H,dhr)
                    kr = apply_rope_flat(self.w_kr[l](a),
                        tensors[f"transformer.h.{l}.path.rope.cos"].cuda(),
                        tensors[f"transformer.h.{l}.path.rope.sin"].cuda(), B,T,Hkv,dhr)
                    Qf,Kf,Vf = build_qkv(qc,qr,kc,kr,v,B,T,H,Hkv,dh,dhr,dq)
                    scale=1.0/math.sqrt(dq)
                    attn=torch.bmm(Qf*scale,Kf.transpose(1,2))
                    mask=torch.triu(torch.ones(T,T,device='cuda')*float('-inf'),diagonal=1)
                    O=torch.bmm(F.softmax(attn+mask.unsqueeze(0),dim=-1),Vf)
                    h = h + self.c_proj[l](O.reshape(B,H,T,dh).transpose(1,2).reshape(B,T,H*dh))
                a2 = self.ln2[l](h)
                h = h + self.mlp_w2[l](F.silu(self.mlp_w1[l](a2))*self.mlp_w3[l](a2))
            return self.lm_head(self.ln_f(h))

    model = Jamba().cuda().eval()
    ids = torch.tensor([[128000, 9906, 11]]).cuda()
    with torch.no_grad():
        logits_pt = model(ids)
    logits_eng = torch.from_numpy(np.fromfile("/tmp/jamba_logits.bin", dtype=np.float32)).cuda().reshape(1,3,-1)
    diff = (logits_pt - logits_eng).abs()
    print(f"\nLogit comparison:")
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
