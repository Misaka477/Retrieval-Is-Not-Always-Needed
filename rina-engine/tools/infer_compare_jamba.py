#!/usr/bin/env python3
"""Compare engine vs PyTorch autoregressive inference for Jamba model."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, json, sys, math, argparse, subprocess

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
    if cos is None or sin is None or cos.shape[-1] < d//2: return x  # SSM layer, no RoPE
    x_f = x.reshape(B*T*H, d)
    half = d // 2
    ct = cos[:T].reshape(T, half).repeat_interleave(H, dim=0)
    st = sin[:T].reshape(T, half).repeat_interleave(H, dim=0)
    x0, x1 = x_f[:, ::2], x_f[:, 1::2]
    return torch.stack([x0*ct-x1*st, x0*st+x1*ct], dim=-1).flatten(-2).reshape(B, T, H*d)

class SSMScan(nn.Module):
    def forward(self, mem, decay):
        B,T,Hdh=mem.shape; H=decay.shape[-1]; dh=Hdh//H
        dv=decay.clamp(min=1e-38)
        lcs=torch.cumsum(dv.log(),dim=1); ca=lcs.exp().clamp(min=1e-30)
        ca_exp=ca.unsqueeze(-1).expand(-1,-1,-1,dh).reshape(B,T,Hdh)
        return ca_exp*torch.cumsum(mem/ca_exp,dim=1)

class JambaInfer(nn.Module):
    def __init__(self, cfg, tensors):
        super().__init__()
        self.cfg = cfg; self.tensors = tensors
        d=cfg['dim']; H=cfg['n_heads']; Hkv=cfg['n_kv_heads']
        dh=cfg['head_dim']; dhr=cfg.get('d_h_r',32); dc=cfg.get('d_c',160)
        ss=cfg.get('ssm_steps',3); L=cfg['n_layers']; V=cfg['vocab_size']
        hd=d*4*2//3//256*256
        self.H=H; self.Hkv=Hkv; self.dh=dh; self.dhr=dhr; self.dc=dc; self.ss=ss; self.dq=dh+dhr
        self.wte = nn.Embedding(V, d)
        self.ln_f = nn.LayerNorm(d, bias=False, eps=1e-5)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
        self.path_ln = nn.ModuleList([nn.LayerNorm(dc, bias=False, eps=1e-5) for _ in range(L)])
        self.mlp_w1 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
        self.mlp_w2 = nn.ModuleList([nn.Linear(hd, d, bias=False) for _ in range(L)])
        self.mlp_w3 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
        self.w_dq = nn.ModuleList([nn.Linear(d, dc, bias=False) for _ in range(L)])
        self.w_mem = nn.ModuleList([nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(ss)]) for _ in range(L)])
        self.w_decay = nn.ModuleList([nn.ModuleList([nn.Linear(dc, H, bias=False) for _ in range(ss)]) for _ in range(L)])
        self.w_out = nn.ModuleList([nn.Linear(d+H*dh, d, bias=False) for _ in range(L)])
        self.w_dqkv = nn.ModuleList([nn.Linear(d, dc, bias=False) for _ in range(L)])
        self.w_uq = nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(L)])
        self.w_uk = nn.ModuleList([nn.Linear(dc, Hkv*dh, bias=False) for _ in range(L)])
        self.w_k2v = nn.ModuleList([nn.Linear(Hkv*dh, Hkv*dh, bias=False) for _ in range(L)])
        self.w_qr = nn.ModuleList([nn.Linear(d, H*dhr, bias=False) for _ in range(L)])
        self.w_kr = nn.ModuleList([nn.Linear(d, Hkv*dhr, bias=False) for _ in range(L)])
        self.c_proj = nn.ModuleList([nn.Linear(H*dh, d, bias=False) for _ in range(L)])
        self.load()
    def load(self):
        for l in range(self.cfg['n_layers']):
            p = f"transformer.h.{l}."
            for name in [f"{p}ln1.weight", f"{p}ln2.weight", f"{p}path.q_norm.weight",
                         f"{p}mlp.w1.weight", f"{p}mlp.w2.weight", f"{p}mlp.w3.weight"]:
                if name in self.tensors:
                    target = {'ln1':self.ln1,'ln2':self.ln2,'q_norm':self.path_ln,
                              'w1':self.mlp_w1,'w2':self.mlp_w2,'w3':self.mlp_w3}[name.split('.')[-2]]
                    target[l].weight.data.copy_(self.tensors[name].float().cuda())
            is_ssm = 'ssm' in self.cfg['layers'][l]['type']
            if is_ssm:
                for k in range(self.ss):
                    self.w_mem[l][k].weight.data.copy_(self.tensors[f"{p}path.w_mem.{k}.weight"].float().cuda())
                    self.w_decay[l][k].weight.data.copy_(self.tensors[f"{p}path.w_decay.{k}.weight"].float().cuda())
                self.w_dq[l].weight.data.copy_(self.tensors[f"{p}path.w_dq.weight"].float().cuda())
                self.w_out[l].weight.data.copy_(self.tensors[f"{p}path.w_out.weight"].float().cuda())
            else:
                self.w_dqkv[l].weight.data.copy_(self.tensors[f"{p}path.w_dqkv.weight"].float().cuda())
                self.w_uq[l].weight.data.copy_(self.tensors[f"{p}path.w_uq.weight"].float().cuda())
                self.w_uk[l].weight.data.copy_(self.tensors[f"{p}path.w_uk.weight"].float().cuda())
                self.w_k2v[l].weight.data.copy_(self.tensors[f"{p}path.w_k2v.weight"].float().cuda())
                self.w_qr[l].weight.data.copy_(self.tensors[f"{p}path.w_qr.weight"].float().cuda())
                self.w_kr[l].weight.data.copy_(self.tensors[f"{p}path.w_kr.weight"].float().cuda())
                self.c_proj[l].weight.data.copy_(self.tensors[f"{p}path.c_proj.weight"].float().cuda())
        self.wte.weight.data.copy_(self.tensors["transformer.wte.weight"].float().cuda())
        self.ln_f.weight.data.copy_(self.tensors["transformer.ln_f.weight"].float().cuda())
        self.lm_head.weight.data.copy_(self.tensors["lm_head.weight"].float().cuda())
    def forward(self, ids):
        B,T = ids.shape
        h = self.wte(ids)
        for l in range(self.cfg['n_layers']):
            is_ssm = 'ssm' in self.cfg['layers'][l]['type']
            a = self.ln1[l](h)
            cq = self.path_ln[l](self.w_dq[l](a) if is_ssm else self.w_dqkv[l](a))
            if is_ssm:
                mems = [self.w_mem[l][k](cq).view(B,T,self.H,self.dh) for k in range(self.ss)]
                dcs = [torch.sigmoid(self.w_decay[l][k](cq)) for k in range(self.ss)]
                d1e=dcs[1].unsqueeze(-1); d2e=dcs[2].unsqueeze(-1)
                ma = mems[0]*d1e*d2e + mems[1]*d2e + mems[2]
                da = dcs[0]*dcs[1]*dcs[2]
                sf = SSMScan()(ma.reshape(B,T,self.H*self.dh), da)
                h = h + self.w_out[l](torch.cat([a, sf], dim=-1))
            else:
                qc = self.w_uq[l](cq); kc = self.w_uk[l](cq); v = self.w_k2v[l](kc)
                cos_q = self.tensors.get(f"transformer.h.{l}.path.rope_q.cos")
                sin_q = self.tensors.get(f"transformer.h.{l}.path.rope_q.sin")
                cos_k = self.tensors.get(f"transformer.h.{l}.path.rope.cos")
                sin_k = self.tensors.get(f"transformer.h.{l}.path.rope.sin")
                qr = apply_rope_flat(self.w_qr[l](a), cos_q.cuda() if cos_q is not None else None,
                    sin_q.cuda() if sin_q is not None else None, B, T, self.H, self.dhr)
                kr = apply_rope_flat(self.w_kr[l](a), cos_k.cuda() if cos_k is not None else None,
                    sin_k.cuda() if sin_k is not None else None, B, T, self.Hkv, self.dhr)
                rep=self.H//self.Hkv; dq=self.dq
                qc_n=qc.reshape(B*T,self.H,self.dh); qr_n=qr.reshape(B*T,self.H,self.dhr)
                kc_n=kc.reshape(B*T,self.Hkv,self.dh); kr_n=kr.reshape(B*T,self.Hkv,self.dhr)
                v_n=v.reshape(B*T,self.Hkv,self.dh)
                Qf=torch.zeros(B*self.H,T,dq,device='cuda'); Kf=torch.zeros(B*self.H,T,dq,device='cuda'); Vf=torch.zeros(B*self.H,T,self.dh,device='cuda')
                for bh in range(B*self.H):
                    b=bh//self.H; hh=bh%self.H; kv=hh//rep
                    for t in range(T):
                        idx=b*T+t
                        Qf[bh,t,:self.dh]=qc_n[idx,hh,:]; Qf[bh,t,self.dh:]=qr_n[idx,hh,:]
                        Kf[bh,t,:self.dh]=kc_n[idx,kv,:]; Kf[bh,t,self.dh:]=kr_n[idx,kv,:]
                        Vf[bh,t,:]=v_n[idx,kv,:]
                scale=1.0/math.sqrt(dq)
                attn=torch.bmm(Qf*scale,Kf.transpose(1,2))
                mask=torch.triu(torch.ones(T,T,device='cuda')*float('-inf'),diagonal=1)
                O=torch.bmm(F.softmax(attn+mask.unsqueeze(0),dim=-1),Vf)
                h = h + self.c_proj[l](O.reshape(B,self.H,T,self.dh).transpose(1,2).reshape(B,T,self.H*self.dh))
            a2 = self.ln2[l](h)
            h = h + self.mlp_w2[l](F.silu(self.mlp_w1[l](a2))*self.mlp_w3[l](a2))
        return self.lm_head(self.ln_f(h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ids', default='128000')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--temp', type=float, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run-engine', action='store_true')
    args = ap.parse_args()

    cfg, tensors = load_rinn_weights(args.model)
    print(f"Model: {cfg['name']} L={cfg['n_layers']} d={cfg['dim']}", file=sys.stderr)

    model = JambaInfer(cfg, tensors).cuda().eval()
    prompt = [int(x) for x in args.ids.split()]
    ids = torch.tensor([prompt]).cuda()
    gen = list(prompt)

    with torch.no_grad():
        for step in range(args.steps):
            logits = model(ids)
            next_logits = logits[0, -1, :]
            if args.temp <= 0:
                next_id = next_logits.argmax().item()
            else:
                probs = F.softmax(next_logits / args.temp, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            gen.append(next_id)
            ids = torch.tensor([gen]).cuda()

    pt_tokens = gen[len(prompt):]
    print(' '.join(str(t) for t in pt_tokens))

    if args.run_engine:
        cmd = ['/home/aquama/Development/RINA_Project/rina-engine/build/rina_infer',
               '--model', args.model, '--ids', args.ids, '--steps', str(args.steps),
               '--temp', str(args.temp)]
        if args.seed and args.temp > 0: cmd += ['--seed', str(args.seed)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        eng_tokens = [int(x) for x in result.stdout.strip().split()]
        print(f"\nComparison:", file=sys.stderr)
        print(f"  PT:  {pt_tokens}", file=sys.stderr)
        print(f"  Eng: {eng_tokens}", file=sys.stderr)
        ok = pt_tokens == eng_tokens
        print(f"  {'✅ MATCH' if ok else '❌ MISMATCH'}", file=sys.stderr)
        if not ok:
            for i,(p,e) in enumerate(zip(pt_tokens,eng_tokens)):
                if p!=e: print(f"  First diff step {i}: PT={p} Eng={e}", file=sys.stderr); break

if __name__ == '__main__':
    main()
