#!/usr/bin/env python3
"""Full alignment test: compare engine vs PyTorch for inference + training.
Usage: python3 align_full.py /tmp/test2.rinn [steps]"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, struct, json, sys, math, subprocess, os

torch.manual_seed(0)
np.random.seed(0)

# ── Load .rinn weights ──
def load_rinn(path):
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
            qtype = idx[pos]; pos += 3  # qtype + block_size
            offset, sz = struct.unpack('<QQ', idx[pos:pos+16]); pos += 16
            f.seek(offset)
            arr = np.frombuffer(f.read(sz), dtype=np.float32)
            tensors[name] = torch.from_numpy(arr.reshape([x for x in shape if x > 0]))
        return cfg, tensors

# ── PyTorch model matching engine architecture ──
class SSMScan(nn.Module):
    def forward(self, mem, decay):
        B, T, Hdh = mem.shape
        H = decay.shape[-1]; dh = Hdh // H
        decay = decay.clamp(min=1e-38)
        ca = torch.cumprod(decay, dim=1)  # [B,T,H]
        ca_exp = ca.unsqueeze(-1).expand(-1, -1, -1, dh)  # [B,T,H,dh]
        mem_r = mem.view(B, T, H, dh)
        wcs = torch.cumsum(mem_r / (ca_exp + 1e-8), dim=1)
        sf = ca_exp * wcs
        return sf.reshape(B, T, H*dh)

class SSMAgg(nn.Module):
    def forward(self, m0, m1, m2, d0, d1, d2):
        B, T, H = d0.shape
        dh_ms = m0.shape[-1] // H
        m0r = m0.view(B, T, H, dh_ms)
        m1r = m1.view(B, T, H, dh_ms)
        m2r = m2.view(B, T, H, dh_ms)
        d1e = d1.unsqueeze(-1)
        d2e = d2.unsqueeze(-1)
        ma = m0r * d1e * d2e + m1r * d2e + m2r
        da = d0 * d1 * d2
        return ma.view(B, T, H*dh_ms), da

def build_model(cfg, tensors):
    d = cfg['dim']; dc = cfg['d_c']; H = cfg['n_heads']; Hkv = cfg['n_kv_heads']
    dh = cfg['head_dim']; dhr = cfg.get('d_h_r', 32); ss = cfg.get('ssm_steps', 3)
    L = cfg['n_layers']; V = cfg['vocab_size']
    hd = d * 4 * 2 // 3 // 256 * 256

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, d, padding_idx=None)
            self.ln_f = nn.LayerNorm(d, bias=False, eps=1e-5)
            self.lm_head = nn.Linear(d, V, bias=False)
            self.ln1 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d, bias=False, eps=1e-5) for _ in range(L)])
            self.path_ln = nn.ModuleList([nn.LayerNorm(dc, bias=False, eps=1e-5) for _ in range(L)])
            self.mlp_w1 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
            self.mlp_w2 = nn.ModuleList([nn.Linear(hd, d, bias=False) for _ in range(L)])
            self.mlp_w3 = nn.ModuleList([nn.Linear(d, hd, bias=False) for _ in range(L)])
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
            # RoPE tables
            self.cos_q = nn.ParameterList(); self.sin_q = nn.ParameterList()
            self.cos_k = nn.ParameterList(); self.sin_k = nn.ParameterList()
            for l in range(L):
                self.cos_q.append(nn.Parameter(torch.zeros(128, dhr//2), requires_grad=False))
                self.sin_q.append(nn.Parameter(torch.zeros(128, dhr//2), requires_grad=False))
                self.cos_k.append(nn.Parameter(torch.zeros(128, dhr//2), requires_grad=False))
                self.sin_k.append(nn.Parameter(torch.zeros(128, dhr//2), requires_grad=False))
            self.load_weights(tensors)
        
        def load_weights(self, sd):
            map_name = {
                'transformer.wte.weight': 'wte.weight',
                'transformer.ln_f.weight': 'ln_f.weight',
                'lm_head.weight': 'lm_head.weight',
            }
            for l in range(L):
                map_name[f'transformer.h.{l}.ln1.weight'] = f'ln1.{l}.weight'
                map_name[f'transformer.h.{l}.ln2.weight'] = f'ln2.{l}.weight'
                map_name[f'transformer.h.{l}.path.q_norm.weight'] = f'path_ln.{l}.weight'
                map_name[f'transformer.h.{l}.mlp.w1.weight'] = f'mlp_w1.{l}.weight'
                map_name[f'transformer.h.{l}.mlp.w2.weight'] = f'mlp_w2.{l}.weight'
                map_name[f'transformer.h.{l}.mlp.w3.weight'] = f'mlp_w3.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_dq.weight'] = f'w_dq.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_dqkv.weight'] = f'w_dqkv.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_uq.weight'] = f'w_uq.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_uk.weight'] = f'w_uk.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_k2v.weight'] = f'w_k2v.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_qr.weight'] = f'w_qr.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_kr.weight'] = f'w_kr.{l}.weight'
                map_name[f'transformer.h.{l}.path.c_proj.weight'] = f'c_proj.{l}.weight'
                map_name[f'transformer.h.{l}.path.w_out.weight'] = f'w_out.{l}.weight'
                map_name[f'transformer.h.{l}.path.rope_q.cos'] = f'cos_q.{l}'
                map_name[f'transformer.h.{l}.path.rope_q.sin'] = f'sin_q.{l}'
                map_name[f'transformer.h.{l}.path.rope.cos'] = f'cos_k.{l}'
                map_name[f'transformer.h.{l}.path.rope.sin'] = f'sin_k.{l}'
                for k in range(ss):
                    map_name[f'transformer.h.{l}.path.w_mem.{k}.weight'] = f'w_mem.{l}.{k}.weight'
                    map_name[f'transformer.h.{l}.path.w_decay.{k}.weight'] = f'w_decay.{l}.{k}.weight'
            
            own = self.state_dict()
            for rinn_name, param in sd.items():
                if rinn_name in map_name:
                    py_name = map_name[rinn_name]
                    if py_name in own:
                        if param.shape == own[py_name].shape:
                            own[py_name].copy_(param.half() if param.dtype==torch.half else param.float())
                        else:
                            print(f"  SHAPE MISMATCH {rinn_name}: {param.shape} vs {own[py_name].shape}")
                    else:
                        print(f"  MISSING key {py_name}")
                elif '.weight' in rinn_name and (rinn_name in own):
                    own[rinn_name].copy_(param.float())
        
        def rope(self, z, cos, sin, T, H):
            """z: [B*T*H, d], cos/sin: [max_T, d//2]"""
            N, D = z.shape
            half = D // 2
            idx_t = torch.arange(N, device=z.device) // H % T
            ct = cos[idx_t]  # [N, half]
            st = sin[idx_t]
            out = torch.zeros_like(z)
            out[:, 0::2] = z[:, 0::2] * ct - z[:, 1::2] * st
            out[:, 1::2] = z[:, 0::2] * st + z[:, 1::2] * ct
            return out
        
        def forward(self, ids):
            B, T = ids.shape
            h = self.wte(ids)
            
            for l in range(L):
                lt = cfg['layers'][l]['type']
                a = h.clone()
                a = self.ln1[l](a)
                
                if 'ssm' in lt:
                    cq = self.w_dq[l](a)
                    cq = self.path_ln[l](cq)
                    mems = [self.w_mem[l][k](cq) for k in range(ss)]
                    decays = [self.w_decay[l][k](cq) for k in range(ss)]
                    decays_sig = [torch.sigmoid(d) for d in decays]
                    ma, da = SSMAgg()(mems[0], mems[1], mems[2], decays_sig[0], decays_sig[1], decays_sig[2])
                    sf = SSMScan()(ma, da)
                    concat = torch.cat([a, sf], dim=-1)
                    a = self.w_out[l](concat)
                else:
                    cq = self.w_dqkv[l](a)
                    cq = self.path_ln[l](cq)
                    qc = self.w_uq[l](cq)       # [B,T,H*dh]
                    kc = self.w_uk[l](cq)        # [B,T,Hkv*dh]
                    v = self.w_k2v[l](kc)         # [B,T,Hkv*dh]
                    qr = self.w_qr[l](a)          # [B,T,H*dhr]
                    kr = self.w_kr[l](a)          # [B,T,Hkv*dhr]
                    # RoPE
                    dq = dh + dhr
                    B_, T_, _ = cq.shape
                    qr_rot = self.rope(qr.view(B_*T_*H, dhr), self.cos_q[l], self.sin_q[l], T_, H)
                    kr_rot = self.rope(kr.view(B_*T_*Hkv, dhr), self.cos_k[l], self.sin_k[l], T_, Hkv)
                    # Build QKV flat [B*H, T, dq] and [B*H, T, dq], matching engine
                    rep = H // Hkv
                    qc_f = qc.view(B_, T_, H, dh).transpose(0,1).reshape(T_, B_*H, dh).permute(1,0,2)  # [Bh,T,dh]
                    qr_f = qr_rot.view(B_, T_, H, dhr).transpose(0,1).reshape(T_, B_*H, dhr).permute(1,0,2)  # [Bh,T,dhr]
                    Qf = torch.cat([qc_f, qr_f], dim=-1)  # [Bh,T,dq]
                    kc_f = kc.view(B_, T_, Hkv, dh).transpose(0,1).reshape(T_, B_*Hkv, dh).permute(1,0,2)
                    kc_e = kc_f.repeat_interleave(rep, dim=0)  # [Bh,T,dh]
                    kr_f = kr_rot.view(B_, T_, Hkv, dhr).transpose(0,1).reshape(T_, B_*Hkv, dhr).permute(1,0,2)
                    kr_e = kr_f.repeat_interleave(rep, dim=0)  # [Bh,T,dhr]
                    Kf = torch.cat([kc_e, kr_e], dim=-1)  # [Bh,T,dq]
                    v_f = v.view(B_, T_, Hkv, dh).transpose(0,1).reshape(T_, B_*Hkv, dh).permute(1,0,2)
                    Vf = v_f.repeat_interleave(rep, dim=0)  # [Bh,T,dh]
                    # Flash attention (causal, matching engine's tiled FA)
                    scale = 1.0 / math.sqrt(dq)
                    Qf = Qf * scale
                    attn = torch.bmm(Qf, Kf.transpose(1,2))  # [Bh,T,T]
                    mask = torch.triu(torch.ones(T_, T_, device=ids.device) * float('-inf'), diagonal=1)
                    attn = attn + mask.unsqueeze(0)
                    P = F.softmax(attn, dim=-1)
                    Of = torch.bmm(P, Vf)  # [Bh,T,dh]
                    O = Of.permute(1,0,2).reshape(T_, B_, H, dh).transpose(0,1).reshape(B_, T_, H*dh)
                    a = self.c_proj[l](O)
                
                h = h + a
                
                # MLP
                a = h.clone()
                a = self.ln2[l](a)
                g = self.mlp_w1[l](a)
                u = self.mlp_w3[l](a)
                a = self.mlp_w2[l](F.silu(g) * u)
                h = h + a
            
            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits
    
    return Model()

def run_engine_infer(rinn_path, ids):
    """Run engine inference, return logits via the test program"""
    # Use test_train's inference mode
    result = subprocess.run(
        ['/home/aquama/Development/RINA_Project/rina-engine/build/test_train',
         '--model', rinn_path, '--steps', '0', '--seq', str(len(ids))],
        capture_output=True, text=True, timeout=30)
    # We can't get logits from the test program easily
    # Use a different approach: call rina_infer directly
    return None

def main():
    rinn_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/test2.rinn'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    cfg, tensors = load_rinn(rinn_path)
    d = cfg['dim']; V = cfg['vocab_size']
    print(f"Model: {cfg['name']} L={cfg['n_layers']} d={d} V={V}")
    
    # Build PyTorch model
    model = build_model(cfg, tensors).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    
    B, T = 1, 32
    
    # ── Inference alignment: compare logits ──
    print("\n═══ Inference Alignment ═══")
    with torch.no_grad():
        model.eval()
        ids = torch.randint(0, V, (B, T), device='cuda')
        pt_logits = model(ids)  # [B,T,V]
        
        # Run engine inference
        import subprocess
        # Create input file
        ids_str = ' '.join(str(int(x)) for x in ids[0])
        result = subprocess.run(
            ['/home/aquama/Development/RINA_Project/rina-engine/build/rina_infer',
             '--model', rinn_path, '--ids', ids_str, '--steps', '1'],
            capture_output=True, text=True, timeout=30)
        print(f"Engine output: {result.stdout.strip()}")
        
        # For proper logit comparison, we'd need the engine to output logits
        # Instead, compare argmax tokens as a proxy
        pt_tok = pt_logits[0, -1].argmax().item()
        print(f"PyTorch argmax: {pt_tok}")
        
        # Check if engine output matches
        if result.returncode == 0 and result.stdout.strip():
            eng_tok = int(result.stdout.strip())
            match = "MATCH" if pt_tok == eng_tok else "MISMATCH"
            print(f"Inference alignment: {match} (PT={pt_tok} Eng={eng_tok})")
        else:
            print("Engine inference: no output or error")
    
    # ── Training alignment: compare loss trajectory ──
    print("\n═══ Training Alignment ═══")
    model.train()
    losses_pt = []
    torch.manual_seed(42)
    
    for step in range(steps):
        data = torch.randint(0, V, (B, T+1), device='cuda')
        x = data[:, :T]
        y = data[:, 1:]
        
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='mean')
        loss.backward()
        opt.step()
        losses_pt.append(loss.item())
        if step < 5 or step % 5 == 4:
            print(f"  PT step {step:3d}: loss={loss.item():.6f}")
    
    print(f"\nPyTorch: initial={losses_pt[0]:.6f} final={losses_pt[-1]:.6f} delta={losses_pt[-1]-losses_pt[0]:+.6f}")
    np.savetxt('/tmp/pt_losses.txt', np.array(losses_pt))
    
    # Get engine losses
    print("\nRunning engine training...")
    result = subprocess.run(
        ['/home/aquama/Development/RINA_Project/rina-engine/build/test_train',
         '--model', rinn_path, '--steps', str(steps), '--seq', str(T)],
        capture_output=True, text=True, timeout=300)
    
    # Parse engine losses
    eng_losses = []
    for line in result.stdout.split('\n'):
        if 'step' in line and 'loss=' in line:
            parts = line.split('loss=')
            if len(parts) >= 2:
                try:
                    val = float(parts[-1].strip())
                    eng_losses.append(val)
                except:
                    pass
    
    if len(eng_losses) > 0 and len(eng_losses) == len(losses_pt):
        avg_diff = np.mean([abs(a-b) for a,b in zip(losses_pt[:len(losses_pt)], eng_losses)])
        max_diff = np.max([abs(a-b) for a,b in zip(losses_pt[:len(losses_pt)], eng_losses)])
        print(f"\n═══ Alignment Results ═══")
        print(f"Losses compared: {len(eng_losses)} steps")
        print(f"  PT   first={losses_pt[0]:.6f} last={losses_pt[-1]:.6f}")
        print(f"  Eng  first={eng_losses[0]:.6f} last={eng_losses[-1]:.6f}")
        print(f"  avg absolute diff: {avg_diff:.6f}")
        print(f"  max absolute diff: {max_diff:.6f}")
        print(f"  delta PT={losses_pt[-1]-losses_pt[0]:+.6f} Eng={eng_losses[-1]-eng_losses[0]:+.6f}")
        ok = avg_diff < 0.1
        print(f"  Alignment: {'OK' if ok else 'NEED WORK'}")
    else:
        print(f"  Engine losses: {len(eng_losses) if eng_losses else 0} (expected {len(losses_pt)})")
        print(f"  Engine stdout: {result.stdout[:500]}")

if __name__ == '__main__':
    main()
