"""RINA-Jamba-DR: 每层双路径 + per-token 动态路由

每个 block 同时包含 SSM 和 Sparse Attention 两条路径，
per-token router 决定走哪条或如何混合。

Training:
  output = α * ssm_out + (1-α) * sparse_out    连续混合
  loss = CE + β * mean(1-α)                      路由惩罚

Inference:
  α > 0.9  → 只算 SSM
  α < 0.1  → 只算 Sparse
  其它     → 混合（极少，1% 以下）
"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F
from rina.model_l3x import SparseIndexManager


def ste_round(x): return x + (x.round() - x).detach()
def q4_0_block(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 7.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-7, 7)
    return (xq * s).reshape(o)
def q2_block(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 1.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-1, 1)
    return (xq * s).reshape(o)
def q1_block(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values
    return (torch.sign(xf) * s).reshape(o)


class Q4BlockLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, weight_bits=4):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim
        self.weight_bits = weight_bits
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.normal_(self.weight, 0, 0.02)
    def forward(self, x):
        w = self.weight
        if self.weight_bits < 16:
            w = q4_0_block(w, 32) if self.weight_bits == 4 else q2_block(w, 32)
        return F.linear(x, w, self.bias)


@dataclass
class DR_Config:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_embd: int = 512
    n_head: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    d_c: int = 128
    d_h_r: int = 32
    dropout: float = 0.0
    bias: bool = False
    use_int4: bool = False
    sparse_k: int = 8
    sparse_window: int = 32
    sparse_local_w: int = 4
    ssm_steps: int = 3
    quant_mode: str = 'q2k_q1v'
    ssm_qbits: int = 4
    weight_bits: int = 4
    routing_penalty: float = 0.01  # 路由熵正则强度（阻止 α 坍缩）


class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__(); self.dim = dim
        inv = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32)/dim))
        self.register_buffer('inv_freq', inv)
        t = torch.arange(max_len, dtype=torch.float32); f = torch.outer(t, inv)
        self.register_buffer('cos', f.cos()); self.register_buffer('sin', f.sin())
    def forward(self, x):
        B,H,T,D = x.shape; h = D//2
        c = self.cos[:T].view(1,1,T,h).expand(B,H,T,h)
        s = self.sin[:T].view(1,1,T,h).expand(B,H,T,h)
        x0, x1 = x[...,::2], x[...,1::2]
        return torch.stack([x0*c-x1*s, x0*s+x1*c], -1).flatten(-2)


class InertiaLayerDR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.d_c = c.d_c; self.n_head = c.n_head; self.d_h = c.head_dim
        self.ssm_steps = getattr(c, 'ssm_steps', 1)
        self.ssm_qbits = getattr(c, 'ssm_qbits', 4)
        wb = getattr(c, 'weight_bits', 4)
        self.w_dq = Q4BlockLinear(c.n_embd, self.d_c, c.bias, wb)
        self.q_norm = nn.LayerNorm(self.d_c, bias=c.bias)
        self.w_mem = nn.ModuleList([Q4BlockLinear(self.d_c, self.n_head*self.d_h, c.bias, wb) for _ in range(self.ssm_steps)])
        self.w_decay = nn.ModuleList([Q4BlockLinear(self.d_c, self.n_head, c.bias, wb) for _ in range(self.ssm_steps)])
        self.w_out = Q4BlockLinear(self.n_head*self.d_h+c.n_embd, c.n_embd, c.bias, wb)
        self.resid_drop = nn.Dropout(c.dropout)
    def _ssm_scan(self, mem, decay, qbits):
        B,T = decay.size(0), decay.size(1)
        d = decay.squeeze(-1) if decay.dim() > 3 else decay
        a = d.unsqueeze(-1).expand(-1,-1,-1,self.d_h)
        lca = torch.cumsum(a.log(), dim=1); ca = lca.exp()
        sf = ca * torch.cumsum(mem/(ca+1e-8), dim=1)
        if qbits:
            for t in [lca, ca, sf]:
                xf = t.float(); s = xf.abs().max()/(2**(qbits-1)-1)
                if s > 1e-10: t.data = (torch.round(xf/s).clamp(-2**(qbits-1),2**(qbits-1)-1)*s).to(t.dtype)
        return sf.reshape(B,T,-1)
    def forward(self, x, k=-1):
        B,T,D = x.shape; cq = self.q_norm(self.w_dq(x))
        mems = [self.w_mem[i](cq).view(B,T,self.n_head,self.d_h) for i in range(self.ssm_steps)]
        decays = [torch.sigmoid(self.w_decay[i](cq)).view(B,T,self.n_head,1) for i in range(self.ssm_steps)]
        d_agg, m_agg = decays[0], mems[0]
        for i in range(1, self.ssm_steps):
            d_agg = d_agg * decays[i]; m_agg = m_agg * decays[i] + mems[i]
        sf = self._ssm_scan(m_agg, d_agg, self.ssm_qbits)
        return self.resid_drop(self.w_out(torch.cat([x, sf], -1))), cq


class MLALayerDR(nn.Module):
    def __init__(self, c, sparse=True):
        super().__init__()
        self.sparse = sparse; self.quant_mode = getattr(c, 'quant_mode', 'q4k_q2v')
        self.n_head = c.n_head; self.n_kv = c.n_kv_heads
        self.n_rep = c.n_head//self.n_kv; self.d = c.n_embd
        self.d_h = c.head_dim; self.d_c = c.d_c; self.d_hr = c.d_h_r
        self.use_qu = c.use_int4; self.sparse_k = c.sparse_k; self.sparse_w = c.sparse_local_w
        wb = getattr(c, 'weight_bits', 4)
        self.w_dqkv = Q4BlockLinear(self.d, self.d_c, c.bias, wb)
        self.q_norm = nn.LayerNorm(self.d_c, bias=c.bias)
        self.k_norm = nn.LayerNorm(self.d_c, bias=c.bias)
        self.w_uq = Q4BlockLinear(self.d_c, self.n_head*self.d_h, c.bias, wb)
        self.w_uk = Q4BlockLinear(self.d_c, self.n_kv*self.d_h, c.bias, wb)
        self.w_k2v = Q4BlockLinear(self.n_kv*self.d_h, self.n_kv*self.d_h, c.bias, wb)
        self.w_qr = Q4BlockLinear(self.d, self.n_head*self.d_hr, c.bias, wb)
        self.w_kr = Q4BlockLinear(self.d, self.n_kv*self.d_hr, c.bias, wb)
        self.rope = RoPE(self.d_hr, c.block_size); self.rope_q = RoPE(self.d_hr, c.block_size)
        self.c_proj = Q4BlockLinear(self.n_head*self.d_h, self.d, c.bias, wb)
        self.attn_drop = nn.Dropout(c.dropout); self.resid_drop = nn.Dropout(c.dropout)
    def forward(self, x, index_mgr=None):
        B,T,_ = x.shape
        cq = self.q_norm(self.w_dqkv(x))
        qc = self.w_uq(cq).view(B,T,self.n_head,self.d_h).transpose(1,2)
        kc = self.w_uk(cq).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        v = self.w_k2v(self.w_uk(cq)).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        if self.use_qu:
            if self.quant_mode == 'q2k_q1v': qc=q2_block(qc); kc=q2_block(kc); v=q1_block(v)
            else: qc=q4_0_block(qc); kc=q4_0_block(kc); v=q2_block(v)
        qr = self.w_qr(x).view(B,T,self.n_head,self.d_hr).transpose(1,2)
        kr = self.w_kr(x).view(B,T,self.n_kv,self.d_hr).transpose(1,2)
        qr,kr = self.rope_q(qr), self.rope(kr)
        if self.n_rep>1:
            kc=kc.repeat_interleave(self.n_rep,1); v=v.repeat_interleave(self.n_rep,1); kr=kr.repeat_interleave(self.n_rep,1)
        q = torch.cat([qc,qr],-1); k = torch.cat([kc,kr],-1)
        if self.sparse and index_mgr is not None and T>1:
            index_mgr.sparse_k=self.sparse_k; index_mgr.local_w=self.sparse_w
            index_mgr.step_forward(cq.detach(),T,x.device); m=index_mgr.get_mask(T)
            if m is None:
                y = F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True)
            else:
                K_per_row=m.sum(-1); K_max=K_per_row.max().item()
                if K_max<1:
                    y = F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True)
                else:
                    v_allow,idx=m.float().topk(K_max,dim=-1); valid=v_allow>0
                    K_use=valid.sum(-1).max().item(); idx=idx[:,:K_use]
                    idx_4d=idx.unsqueeze(0).unsqueeze(0).expand(-1,self.n_head,-1,-1)
                    k_sel=torch.gather(k.unsqueeze(2).expand(-1,-1,T,-1,-1),3,
                        idx_4d.unsqueeze(-1).expand(-1,-1,-1,-1,k.size(-1)))
                    v_sel=torch.gather(v.unsqueeze(2).expand(-1,-1,T,-1,-1),3,
                        idx_4d.unsqueeze(-1).expand(-1,-1,-1,-1,v.size(-1)))
                    y=F.scaled_dot_product_attention(q.reshape(-1,1,1,k.size(-1)),
                        k_sel.reshape(-1,1,K_use,k.size(-1)),
                        v_sel.reshape(-1,1,K_use,v.size(-1)),dropout_p=0,is_causal=False)
                    y=y.squeeze(1).reshape(B,self.n_head,T,-1).transpose(1,2).contiguous().view(B,T,-1)
                    return self.resid_drop(self.c_proj(y)), cq
        else:
            y = F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.c_proj(y)), cq


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__(); h = c.n_embd*4*2//3//256*256; wb = getattr(c, 'weight_bits', 4)
        self.w1 = Q4BlockLinear(c.n_embd, h, c.bias, wb)
        self.w2 = Q4BlockLinear(h, c.n_embd, c.bias, wb)
        self.w3 = Q4BlockLinear(c.n_embd, h, c.bias, wb)
        self.dp = nn.Dropout(c.dropout)
    def forward(self, x): return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))


class DRBlock(nn.Module):
    """每层双路径 + per-token 质量感知路由（轻量版）"""
    def __init__(self, c, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.ln2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        # router: 输出 α ∈ (0,1)，0=Sparse, 1=SSM
        self.router = Q4BlockLinear(c.n_embd, 1, bias=True, weight_bits=c.weight_bits)
        self.ssm_path = InertiaLayerDR(c)
        self.sparse_path = MLALayerDR(c, sparse=True)
        self.mlp = SwiGLU(c)

    def forward(self, x, index_mgr=None):
        ln = self.ln1(x)

        ssm_out, _ = self.ssm_path(ln, k=-1)
        sparse_out, _ = self.sparse_path(ln, index_mgr)

        # router: per-token → sigmoid
        logit = self.router(ln)  # [B, T, 1]
        alpha = torch.sigmoid(logit)

        # 混合
        out = alpha * ssm_out + (1 - alpha) * sparse_out
        r = x + out
        return r + self.mlp(self.ln2(r)), alpha


class RINA_Jamba_DR(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.routing_penalty = getattr(c, 'routing_penalty', 0.1)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([DRBlock(c, i) for i in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd, bias=c.bias),
        ))
        self.lm_head = Q4BlockLinear(c.n_embd, c.vocab_size, bias=c.bias, weight_bits=c.weight_bits)
        self.transformer.wte.weight = self.lm_head.weight
        core = sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()
        print(f'RINA_Jamba_DR: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')

    def forward(self, idx, targets=None, sparse_mgr=None):
        B,T = idx.size(); assert T <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        total_alpha = 0.0
        total_quality_bonus = 0.0
        n_layers = 0
        sparse_mgr = None  # 每层独立管理
        for b in self.transformer.h:
            x, alpha = b(x, sparse_mgr)
            total_alpha = total_alpha + alpha.mean()
            n_layers += 1
        x = self.transformer.ln_f(x)
        l = self.lm_head(x)
        if targets is not None:
            ce = F.cross_entropy(l.view(-1,l.size(-1)), targets.reshape(-1), ignore_index=-1)
            alpha_mean = total_alpha / n_layers
            # 路由熵正则：防止 α 坍缩到 0 或 1，鼓励探索两条路径
            # H(α) = -α·log(α) - (1-α)·log(1-α)
            eps_a = alpha_mean.clamp(1e-7, 1-1e-7)
            entropy = -(eps_a * eps_a.log() + (1 - eps_a) * (1 - eps_a).log())
            total_loss = ce - self.routing_penalty * entropy  # -entropy because we MAX entropy
            return l, total_loss, ce, alpha_mean
        return l[:,[-1],:], None


def test():
    """小规模验证"""
    cfg = DR_Config(n_embd=128, n_layer=4, n_head=4, n_kv_heads=2, d_c=32, vocab_size=256)
    model = RINA_Jamba_DR(cfg)
    x = torch.randint(0, 256, (2, 32))
    l, loss, ce, alpha = model(x, x)
    print(f'  Forward OK: loss={loss.item():.3f} ce={ce.item():.3f} alpha={alpha.item():.3f}')
    print(f'  α={alpha.item():.3f} → {"SSM dominant" if alpha > 0.5 else "Sparse dominant"}')
    return model

if __name__ == '__main__':
    test()
