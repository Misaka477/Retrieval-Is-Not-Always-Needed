"""RINA — Retrieval Is Not Always Needed: MLA + K→V + GQA + RoPE + SwiGLU + 1.58-bit + int4/2。"""
import math, inspect
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F

# ── 配置 ──

@dataclass
class RINAConfig:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12
    n_head: int = 8; n_kv_heads: int = 4; n_embd: int = 512
    head_dim: int = 64; d_c: int = 128; d_h_r: int = 32
    dropout: float = 0.0; bias: bool = False
    use_rope: bool = True; use_swiglu: bool = True; use_k2v: bool = True
    use_158: bool = False  # 1.58-bit 三元量化权重
    use_int4: bool = False  # 训练时 int4/2 量化模拟

# ── STE 量化工具 ──

def ste_round(x):
    """STE 近似取整：前向取整，反向梯度直通。"""
    return x + (x.round() - x).detach()

def q4(x, gs=32):
    """int4 分组量化 + STE。[-7,7]，每 gs 个元素共享一个 scale。"""
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 7.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-7, 7)
    return (xq * s).reshape(o)

def q2(x, gs=32):
    """int2 分组量化 + STE。{-1,0,+1}，每 gs 个元素共享一个 scale。"""
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 1.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-1, 1)
    return (xq * s).reshape(o)

# ── 1.58-bit 三元量化线性层 ──

def lin(c, in_f, out_f, bias=None):
    b = bias if bias is not None else c.bias
    if isinstance(c, bool):  # 兼容直接传 use_158
        return BitLinear(in_f, out_f, b) if c else nn.Linear(in_f, out_f, b)
    return BitLinear(in_f, out_f, b) if getattr(c, 'use_158', False) else nn.Linear(in_f, out_f, b)

class BitLinear(nn.Module):
    """1.58-bit 三元权重 + STE：前向 {-1,0,+1}×scale，反向 fp32 梯度直通。"""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.weight, 0.0, 0.02)
        g = max(1, out_f // 128)
        while out_f % g != 0 and g > 1:
            g -= 1
        self.groups = g
        self.scale = nn.Parameter(torch.ones(self.groups))
    def forward(self, x):
        w = self.weight; g = self.groups
        w_g = w.view(g, -1)
        s = w_g.abs().mean(dim=-1, keepdim=True) * self.scale.view(g, 1)
        w_q = torch.clamp(torch.round(w_g / (s + 1e-8)), -1, 1) * s
        w_q = w_q.view_as(w)
        w_e = w + (w_q - w).detach()
        return F.linear(x, w_e, self.bias)

# ── RoPE ──

class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__(); self.dim = dim
        inv = 1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
        self.register_buffer('inv_freq', inv)
        t = torch.arange(max_len, dtype=torch.float32)
        f = torch.outer(t, inv)
        self.register_buffer('cos', f.cos()); self.register_buffer('sin', f.sin())
    def forward(self, x):
        B,H,T,D=x.shape; h=D//2
        c=self.cos[:T].view(1,1,T,h).expand(B,H,T,h)
        s=self.sin[:T].view(1,1,T,h).expand(B,H,T,h)
        x0,x1=x[...,::2],x[...,1::2]
        return torch.stack([x0*c-x1*s, x0*s+x1*c], dim=-1).flatten(-2)

# ── MLA 核心层 ──

class MLALayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head=c.n_head; self.n_kv=c.n_kv_heads; self.n_rep=c.n_head//self.n_kv
        self.d=c.n_embd; self.d_h=c.head_dim; self.d_c=c.d_c; self.d_hr=c.d_h_r
        self.use_qu=c.use_int4
        self.w_dqkv=lin(c, self.d, self.d_c)
        self.q_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.k_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.w_uq=lin(c, self.d_c, self.n_head*self.d_h)
        self.w_uk=lin(c, self.d_c, self.n_kv*self.d_h)
        self.w_k2v=lin(c, self.n_kv*self.d_h, self.n_kv*self.d_h)
        self.w_qr=lin(c, self.d, self.n_head*self.d_hr)
        self.w_kr=lin(c, self.d, self.n_kv*self.d_hr)
        self.rope=RoPE(self.d_hr, c.block_size)
        self.rope_q=RoPE(self.d_hr, c.block_size)
        self.c_proj=lin(c, self.n_head*self.d_h, self.d)
        self.attn_drop=nn.Dropout(c.dropout); self.resid_drop=nn.Dropout(c.dropout)
        self.flash=hasattr(F,'scaled_dot_product_attention')
    def forward(self, x):
        B,T,_=x.shape; cq=self.q_norm(self.w_dqkv(x))
        qc=self.w_uq(cq).view(B,T,self.n_head,self.d_h).transpose(1,2)
        kc=self.w_uk(cq).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        v=self.w_k2v(self.w_uk(cq)).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        if self.use_qu:
            qc = q4(qc); kc = q4(kc); v = q2(v)
        qr=self.w_qr(x).view(B,T,self.n_head,self.d_hr).transpose(1,2)
        kr=self.w_kr(x).view(B,T,self.n_kv,self.d_hr).transpose(1,2)
        qr,kr=self.rope_q(qr),self.rope(kr)
        if self.n_rep>1:
            kc=kc.repeat_interleave(self.n_rep,1);v=v.repeat_interleave(self.n_rep,1);kr=kr.repeat_interleave(self.n_rep,1)
        q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
        if self.flash:
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True)
        else:
            a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
            m=torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
            a=a.masked_fill(m==0,float('-inf'));a=F.softmax(a,-1);a=self.attn_drop(a);y=a@v
        y=y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.c_proj(y))

# ── SwiGLU FFN ──

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.n_embd*4*2//3//256*256
        self.w1=lin(c, c.n_embd, h); self.w2=lin(c, h, c.n_embd); self.w3=lin(c, c.n_embd, h)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))

class Block(nn.Module):
    def __init__(self,c):super().__init__();self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias);self.attn=MLALayer(c);self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias);self.mlp=SwiGLU(c) if c.use_swiglu else None
    def forward(self,x):x=x+self.attn(self.ln1(x));x=x+self.mlp(self.ln2(x));return x

# ── RINA 模型 ──

class RINA(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd, bias=c.bias),
        ))
        self.lm_head = lin(c, c.n_embd, c.vocab_size)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._iw)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))
        core = sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()
        print(f'RINA: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')

    def _iw(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size(); assert T <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        for b in self.transformer.h: x = b(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            l = self.lm_head(x)
            loss = F.cross_entropy(l.view(-1, l.size(-1)), targets.reshape(-1), ignore_index=-1)
            return l, loss
        return self.lm_head(x[:, [-1], :]), None

    def configure_optimizers(self, wd, lr, betas, dt):
        p = {n: p for n, p in self.named_parameters() if p.requires_grad}
        d = [p for n, p in p.items() if p.dim() >= 2]
        n_ = [p for n, p in p.items() if p.dim() < 2]
        return torch.optim.AdamW([
            {'params': d, 'weight_decay': wd},
            {'params': n_, 'weight_decay': 0.0}
        ], lr=lr, betas=betas, fused='fused' in inspect.signature(torch.optim.AdamW).parameters)

    @torch.no_grad()
    def generate(self, idx, max_new, temp=1.0, top_k=None):
        for _ in range(max_new):
            c = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            l, _ = self(c); l = l[:, -1, :] / temp
            if top_k is not None:
                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                l[l < v[:, [-1]]] = float('-inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(l, -1), 1)), 1)
        return idx
