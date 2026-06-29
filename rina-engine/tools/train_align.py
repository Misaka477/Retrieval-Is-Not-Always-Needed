#!/usr/bin/env python3
"""100-step training alignment: run PyTorch reference and compare with engine."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, subprocess, struct, os, sys, json, tempfile, math

torch.manual_seed(0)
np.random.seed(0)

# ── Load model from .rinn ──
def load_rinn_tensors(path):
    """Load all tensor data from .rinn as numpy arrays."""
    with open(path, 'rb') as f:
        hdr = f.read(512)
        n_tensors = struct.unpack('<I', hdr[8:12])[0]
        idx_off, idx_sz = struct.unpack('<QQ', hdr[32:48])
        f.seek(idx_off); idx = f.read(idx_sz)
        config_len = struct.unpack('<Q', hdr[24:32])[0]
        f.seek(512); cfg = json.loads(f.read(config_len))
        
        pos = 0; tensors = {}
        while pos < idx_sz:
            nl = struct.unpack('<H', idx[pos:pos+2])[0]; pos += 2
            name = idx[pos:pos+nl].decode(); pos += nl
            shape = struct.unpack('<iiii', idx[pos:pos+16]); pos += 16
            qtype = idx[pos]; pos += 1  # 6=FP32
            blk = struct.unpack('<H', idx[pos:pos+2])[0]; pos += 2
            offset, sz = struct.unpack('<QQ', idx[pos:pos+16]); pos += 16
            f.seek(offset)
            data = np.frombuffer(f.read(sz), dtype=np.float32).copy()
            s = [x for x in shape if x > 0]
            tensors[name] = torch.from_numpy(data.reshape(s))
        return cfg, tensors

# ── Build PyTorch model matching engine architecture ──
class SSMStep(nn.Module):
    def __init__(self, H, dh):
        super().__init__()
        self.H, self.dh = H, dh
    def forward(self, mem_k, decay_k):
        # sigmoid on decay
        decay = torch.sigmoid(decay_k)
        B, T = decay.shape[0], decay.shape[1]
        ca = torch.cumprod(decay, dim=1).clamp(min=1e-38)  # [B,T,H]
        ca_exp = ca.unsqueeze(-1).expand(-1, -1, -1, self.dh)
        ma = mem_k
        wcs = torch.cumsum(ma / (ca_exp + 1e-8), dim=1)
        sf = ca_exp * wcs
        return sf, ca

class SSMBlock(nn.Module):
    def __init__(self, d, dc, H, dh, ssm_steps=3):
        super().__init__()
        self.ssm_steps = ssm_steps
        self.w_dq = nn.Linear(d, dc, bias=False)
        self.q_norm = nn.LayerNorm(dc, eps=1e-5)
        self.w_mem = nn.ModuleList([nn.Linear(dc, H*dh, bias=False) for _ in range(ssm_steps)])
        self.w_decay = nn.ModuleList([nn.Linear(dc, H, bias=False) for _ in range(ssm_steps)])
        self.ssm = SSMStep(H, dh)
        self.w_out = nn.Linear(d + H*dh, d, bias=False)
    
    def forward(self, x):
        cq = self.w_dq(x)
        cq = self.q_norm(cq)
        mems, decays = [], []
        for k in range(self.ssm_steps):
            mems.append(self.w_mem[k](cq))
            decays.append(self.w_decay[k](cq))
        da = decays[0] * decays[1] * decays[2]
        # agg: combine mems
        B, T, H, dh = x.shape[0], x.shape[1], self.w_mem[0].weight.shape[0] // self.w_mem[0].weight.shape[1], 0
        # Actually compute shapes dynamically
        hd_ = self.w_mem[0].weight.shape[0]
        dh_ = hd_ // H
        ma = mems[0] * decays[1].unsqueeze(-1) * decays[2].unsqueeze(-1) \
           + mems[1] * decays[2].unsqueeze(-1) \
           + mems[2]
        # rewrite: ma has shape [B,T,H,dh], need to decompose
        ma = mems[0].view(B,T,H,dh_) * decays[1].unsqueeze(-1) * decays[2].unsqueeze(-1) \
           + mems[1].view(B,T,H,dh_) * decays[2].unsqueeze(-1) \
           + mems[2].view(B,T,H,dh_)
        ma = ma.view(B,T,H*dh_)
        # combine da
        da_comb = decays[0] * decays[1] * decays[2]
        # scan
        sf, _ = self.ssm(ma, da_comb)
        concat = torch.cat([x, sf], dim=-1)
        return self.w_out(concat)

class MLABlock(nn.Module):
    def __init__(self, d, dc, H, Hkv, dh, dhr):
        super().__init__()
        dq = dh + dhr
        self.w_dqkv = nn.Linear(d, dc, bias=False)
        self.q_norm = nn.LayerNorm(dc, eps=1e-5)
        self.w_uq = nn.Linear(dc, H*dh, bias=False)
        self.w_uk = nn.Linear(dc, Hkv*dh, bias=False)
        self.w_k2v = nn.Linear(Hkv*dh, Hkv*dh, bias=False)
        self.w_qr = nn.Linear(d, H*dhr, bias=False)
        self.w_kr = nn.Linear(d, Hkv*dhr, bias=False)
        self.c_proj = nn.Linear(H*dh, d, bias=False)
        self.H, self.Hkv, self.dh, self.dhr, self.dq = H, Hkv, dh, dhr, dq
    
    def forward(self, x, cos_q, sin_q, cos_k, sin_k):
        B, T, d = x.shape
        cq = self.w_dqkv(x)
        cq = self.q_norm(cq)
        qc = self.w_uq(cq)
        kc = self.w_uk(cq)
        v = self.w_k2v(kc)
        qr = self.w_qr(x)
        kr = self.w_kr(x)
        # RoPE
        def rope(z, cos, sin):
            B, T, H, d_dim = z.shape
            half = d_dim // 2
            z = z.view(B*T*H, d_dim)
            idx_t = torch.arange(B*T, device=z.device) // 1 * 0  # simplified
            # Actually compute correctly
            z_even = z[:, 0::2]  # [BTH, half]
            z_odd = z[:, 1::2]
            cos_t = cos[:T].repeat_interleave(H, dim=0)  # [BTH, half]
            sin_t = sin[:T].repeat_interleave(H, dim=0)
            out = torch.zeros_like(z)
            out[:, 0::2] = z_even * cos_t - z_odd * sin_t
            out[:, 1::2] = z_even * sin_t + z_odd * cos_t
            return out.view(B, T, H, d_dim)
        # Build QKV
        qc = qc.view(B, T, self.H, self.dh)
        qr = qr.view(B, T, self.H, self.dhr)
        kc = kc.view(B, T, self.Hkv, self.dh)
        kr = kr.view(B, T, self.Hkv, self.dhr)
        v = v.view(B, T, self.Hkv, self.dh)
        
        Q = torch.cat([qc, rope(qr, cos_q, sin_q)], dim=-1)  # [B,T,H,dq]
        K = torch.cat([kc, rope(kr, cos_k, sin_k)], dim=-1)  # [B,T,Hkv,dq]
        V = v  # [B,T,Hkv,dh]
        
        # Flash attention: causal
        scale = 1.0 / math.sqrt(self.dq)
        Q = Q * scale
        attn = torch.matmul(Q, K.transpose(-2, -1))  # [B,T,H,T]
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        O = torch.matmul(attn, V)  # [B,T,H,dh]
        O = O.view(B, T, self.H * self.dh)
        return self.c_proj(O)

class MLPBlock(nn.Module):
    def __init__(self, d, hd):
        super().__init__()
        self.w1 = nn.Linear(d, hd, bias=False)
        self.w2 = nn.Linear(hd, d, bias=False)
        self.w3 = nn.Linear(d, hd, bias=False)
    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(F.silu(gate) * up)

class SimpleEngine(nn.Module):
    def __init__(self, cfg, state_dict):
        super().__init__()
        d = cfg['dim']
        self.wte = nn.Linear(d, d, bias=False)  # placeholder, will load weights
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.layers = nn.ModuleList()
        for l in range(cfg['n_layers']):
            lt = cfg['layers'][l]['type']
            if 'ssm' in lt:
                self.layers.append(SSMBlock(d, cfg['d_c'], cfg['n_heads'], cfg['head_dim'], cfg['ssm_steps']))
            else:
                self.layers.append(MLABlock(d, cfg['d_c'], cfg['n_heads'], cfg['n_kv_heads'], cfg['head_dim'], cfg.get('d_h_r', 32)))
            # Add LN1, LN2, MLP for each layer
            setattr(self, f'ln1_{l}', nn.LayerNorm(d, eps=1e-5))
            setattr(self, f'ln2_{l}', nn.LayerNorm(d, eps=1e-5))
            setattr(self, f'mlp_{l}', MLPBlock(d, d*4*2//3//256*256))
        self.lm_head = nn.Linear(d, cfg['vocab_size'], bias=False)
        self.load_weights(state_dict)
    
    def load_weights(self, sd):
        """Copy weights from loaded state dict to model."""
        # Simple approach: use the state dict to set weights directly
        own = self.state_dict()
        for name, param in sd.items():
            if name in own:
                if param.shape == own[name].shape:
                    own[name].copy_(param.to(own[name].dtype))
                else:
                    print(f"  Shape mismatch: {name}: {param.shape} vs {own[name].shape}")
    
    def forward(self, ids, targets=None):
        B, T = ids.shape
        h = self.wte.weight[ids]  # embedding
        for l in range(len(self.layers)):
            ln1 = getattr(self, f'ln1_{l}')
            ln2 = getattr(self, f'ln2_{l}')
            mlp = getattr(self, f'mlp_{l}')
            
            a = h
            a = ln1(a)
            out = self.layers[l](a)
            h = h + out
            
            a = h
            a = ln2(a)
            a = mlp(a)
            h = h + a
        
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='mean')
            return logits, loss
        return logits

# ── Main ──
def main():
    rinn_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/test2.rinn'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Load model
    cfg, tensors = load_rinn_tensors(rinn_path)
    print(f"Model: {cfg['name']} L={cfg['n_layers']} d={cfg['dim']} V={cfg['vocab_size']}")
    
    model = SimpleEngine(cfg, tensors).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    
    B, T = 1, 32
    V = cfg['vocab_size']
    losses_pt = []
    
    # Save initial weights for engine comparison
    initial_wt = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    for step in range(steps):
        data = torch.randint(0, V, (B, T+1), device='cuda')
        ids = data[:, :T]
        targets = data[:, 1:]
        
        opt.zero_grad()
        logits, loss = model(ids, targets)
        loss.backward()
        opt.step()
        
        losses_pt.append(loss.item())
        if step < 5 or step % 10 == 9:
            print(f"  PT step {step:3d}: loss={loss.item():.6f}")
    
    print(f"\nPyTorch {steps} steps: initial={losses_pt[0]:.6f} final={losses_pt[-1]:.6f} delta={losses_pt[0]-losses_pt[-1]:.6f}")
    
    # Save losses for comparison
    np.savetxt('/tmp/pt_losses.txt', np.array(losses_pt))
    print("Saved: /tmp/pt_losses.txt")

if __name__ == '__main__':
    main()
