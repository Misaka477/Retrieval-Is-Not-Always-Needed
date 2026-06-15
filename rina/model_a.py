"""RINA Route A — Latent Indexed Attention: 全量 attention 训练，latent 对比学习为稀疏索引做准备"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F

def ste_round(x):
    return x + (x.round() - x).detach()

def q4(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 7.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-7, 7)
    return (xq * s).reshape(o)

def q2(x, gs=32):
    o = x.shape; xf = x.reshape(-1, gs)
    s = xf.abs().max(-1, keepdim=True).values / 1.0
    xq = ste_round(xf / (s + 1e-8)).clamp(-1, 1)
    return (xq * s).reshape(o)

@dataclass
class RINA_A_Config:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12
    n_head: int = 8; n_kv_heads: int = 4; n_embd: int = 512
    head_dim: int = 64; d_c: int = 128; d_h_r: int = 32
    dropout: float = 0.0; bias: bool = False
    use_rope: bool = True; use_swiglu: bool = True; use_k2v: bool = True
    use_int4: bool = False

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

class MLALayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head=c.n_head; self.n_kv=c.n_kv_heads; self.n_rep=c.n_head//self.n_kv
        self.d=c.n_embd; self.d_h=c.head_dim; self.d_c=c.d_c; self.d_hr=c.d_h_r
        self.use_qu=c.use_int4
        self.w_dqkv=nn.Linear(self.d, self.d_c, bias=c.bias)
        self.q_norm=nn.LayerNorm(self.d_c, bias=c.bias)
        self.k_norm=nn.LayerNorm(self.d_c, bias=c.bias)
        self.w_uq=nn.Linear(self.d_c, self.n_head*self.d_h, bias=c.bias)
        self.w_uk=nn.Linear(self.d_c, self.n_kv*self.d_h, bias=c.bias)
        self.w_k2v=nn.Linear(self.n_kv*self.d_h, self.n_kv*self.d_h, bias=c.bias)
        self.w_qr=nn.Linear(self.d, self.n_head*self.d_hr, bias=c.bias)
        self.w_kr=nn.Linear(self.d, self.n_kv*self.d_hr, bias=c.bias)
        self.rope=RoPE(self.d_hr, c.block_size)
        self.rope_q=RoPE(self.d_hr, c.block_size)
        self.c_proj=nn.Linear(self.n_head*self.d_h, self.d, bias=c.bias)
        self.attn_drop=nn.Dropout(c.dropout); self.resid_drop=nn.Dropout(c.dropout)
        self.flash=hasattr(F,'scaled_dot_product_attention')
    def forward(self, x):
        B,T,_=x.shape
        cq=self.q_norm(self.w_dqkv(x))
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
        return self.resid_drop(self.c_proj(y)), cq

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.n_embd*4*2//3//256*256
        self.w1=nn.Linear(c.n_embd,h,bias=c.bias);self.w2=nn.Linear(h,c.n_embd,bias=c.bias);self.w3=nn.Linear(c.n_embd,h,bias=c.bias)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))

class Block(nn.Module):
    def __init__(self,c):super().__init__();self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias);self.attn=MLALayer(c);self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias);self.mlp=SwiGLU(c)
    def forward(self,x):a,lat=self.attn(self.ln1(x));x=x+a;x=x+self.mlp(self.ln2(x));return x,lat

class RINA_A(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd, bias=c.bias),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=c.bias)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._iw)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))
        core = sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()
        print(f'RINA_A: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')

    def _iw(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size(); assert T <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        lats = []
        for b in self.transformer.h:
            x, lat = b(x)
            lats.append(lat)
        x = self.transformer.ln_f(x)
        if targets is not None:
            l = self.lm_head(x)
            loss = F.cross_entropy(l.view(-1, l.size(-1)), targets.reshape(-1), ignore_index=-1)
            return l, loss, torch.stack(lats, dim=1)
        return self.lm_head(x[:, [-1], :]), None, None

    def compute_contrastive_loss(self, lats, idx=None, margin=0.3):
        """Triplet margin loss: 相邻 token 吸引, 远距 token 排斥, 防止全局坍缩"""
        B, L, T, D = lats.shape
        lats = lats[:, 1:].mean(dim=1)       # [B, T, D], skip 1st CE-heavy layer
        lats = F.normalize(lats, dim=-1)

        pos_sim = (lats[:, :-1] * lats[:, 1:]).sum(-1).mean()      # 邻接吸引
        shift = T // 2
        neg_sim = (lats[:, :-shift] * lats[:, shift:]).sum(-1).mean()  # 远距排斥

        return F.relu(margin - (pos_sim - neg_sim))
