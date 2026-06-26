"""RINA AC — 混合架构：浅层惯性波 (C) + 深层注意力索引 (A)，类 Jamba"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F

# ── STE 量化工具 ──
def ste_round(x): return x + (x.round() - x).detach()
def q4(x, gs=32):
    o=x.shape;xf=x.reshape(-1,gs);s=xf.abs().max(-1,keepdim=True).values/7.0
    xq=ste_round(xf/(s+1e-8)).clamp(-7,7);return (xq*s).reshape(o)
def q2(x, gs=32):
    o=x.shape;xf=x.reshape(-1,gs);s=xf.abs().max(-1,keepdim=True).values/1.0
    xq=ste_round(xf/(s+1e-8)).clamp(-1,1);return (xq*s).reshape(o)

@dataclass
class RINA_AC_Config:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12; n_embd: int = 512
    n_head: int = 8; n_kv_heads: int = 4; head_dim: int = 64; d_c: int = 128; d_h_r: int = 32
    dropout: float = 0.0; bias: bool = False; use_int4: bool = False
    inertia_layers: tuple = (0, 1, 2, 3)     # 惯性波层
    attention_layers: tuple = (4, 5, 6, 7)   # 全量 attention
    sparse_layers: tuple = (8, 9, 10, 11)    # 稀疏索引 attention
    sparse_k: int = 8; sparse_local_w: int = 4

# ── RoPE ──
class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__(); self.dim=dim
        inv=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
        self.register_buffer('inv_freq',inv)
        t=torch.arange(max_len,dtype=torch.float32);f=torch.outer(t,inv)
        self.register_buffer('cos',f.cos());self.register_buffer('sin',f.sin())
    def forward(self,x):
        B,H,T,D=x.shape;h=D//2
        c=self.cos[:T].view(1,1,T,h).expand(B,H,T,h)
        s=self.sin[:T].view(1,1,T,h).expand(B,H,T,h)
        x0,x1=x[...,::2],x[...,1::2]
        return torch.stack([x0*c-x1*s,x0*s+x1*c],dim=-1).flatten(-2)

# ── C: Inertia Wave ──
class InertiaLayer(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.d_c=c.d_c;self.n_head=c.n_head;self.d_h=c.head_dim
        self.w_dq=nn.Linear(c.n_embd,self.d_c,bias=c.bias)
        self.q_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.w_mem=nn.Linear(self.d_c,self.n_head*self.d_h,bias=c.bias)
        self.w_decay=nn.Linear(self.d_c,self.n_head,bias=c.bias)
        self.w_out=nn.Linear(self.n_head*self.d_h+c.n_embd,c.n_embd,bias=c.bias)
        self.resid_drop=nn.Dropout(c.dropout)
    def forward(self,x):
        B,T,D=x.shape;cq=self.q_norm(self.w_dq(x))
        mem=self.w_mem(cq).view(B,T,self.n_head,self.d_h)
        decay=torch.sigmoid(self.w_decay(cq))
        a=decay.unsqueeze(-1).expand(-1,-1,-1,self.d_h)
        ca=torch.cumprod(a,dim=1)
        states=ca*torch.cumsum(mem/(ca+1e-8),dim=1)
        return self.resid_drop(self.w_out(torch.cat([x,states.reshape(B,T,-1)],-1)))

# ── A: MLA Attention ──
class MLALayer(nn.Module):
    def __init__(self,c,sparse=False):
        super().__init__()
        self.sparse=sparse;self.n_head=c.n_head;self.n_kv=c.n_kv_heads
        self.n_rep=c.n_head//self.n_kv;self.d=c.n_embd;self.d_h=c.head_dim
        self.d_c=c.d_c;self.d_hr=c.d_h_r;self.use_qu=c.use_int4
        self.sparse_k=c.sparse_k;self.sparse_w=c.sparse_local_w
        self.w_dqkv=nn.Linear(self.d,self.d_c,bias=c.bias)
        self.q_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.k_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.w_uq=nn.Linear(self.d_c,self.n_head*self.d_h,bias=c.bias)
        self.w_uk=nn.Linear(self.d_c,self.n_kv*self.d_h,bias=c.bias)
        self.w_k2v=nn.Linear(self.n_kv*self.d_h,self.n_kv*self.d_h,bias=c.bias)
        self.w_qr=nn.Linear(self.d,self.n_head*self.d_hr,bias=c.bias)
        self.w_kr=nn.Linear(self.d,self.n_kv*self.d_hr,bias=c.bias)
        self.rope=RoPE(self.d_hr,c.block_size)
        self.rope_q=RoPE(self.d_hr,c.block_size)
        self.c_proj=nn.Linear(self.n_head*self.d_h,self.d,bias=c.bias)
        self.attn_drop=nn.Dropout(c.dropout);self.resid_drop=nn.Dropout(c.dropout)
    def forward(self,x,latents=None):
        B,T,_=x.shape
        cq=self.q_norm(self.w_dqkv(x))
        qc=self.w_uq(cq).view(B,T,self.n_head,self.d_h).transpose(1,2)
        kc=self.w_uk(cq).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        v=self.w_k2v(self.w_uk(cq)).view(B,T,self.n_kv,self.d_h).transpose(1,2)
        if self.use_qu: qc=q4(qc);kc=q4(kc);v=q2(v)
        qr=self.w_qr(x).view(B,T,self.n_head,self.d_hr).transpose(1,2)
        kr=self.w_kr(x).view(B,T,self.n_kv,self.d_hr).transpose(1,2)
        qr,kr=self.rope_q(qr),self.rope(kr)
        if self.n_rep>1:
            kc=kc.repeat_interleave(self.n_rep,1);v=v.repeat_interleave(self.n_rep,1);kr=kr.repeat_interleave(self.n_rep,1)
        q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
        if self.sparse and latents is not None and T>1:
            lnew=F.normalize(cq,dim=-1).detach()
            lats=torch.cat([latents,lnew],dim=1)
            lats=F.normalize(lats.mean(0,keepdim=True).expand(T,-1,-1),dim=-1) if latents.dim()==3 else latents
            sim=torch.bmm(F.normalize(cq,-1),F.normalize(cq,-1).transpose(1,2))
            _,idx=torch.topk(sim.squeeze(0),min(self.sparse_k,T),dim=-1)
            allow=torch.zeros(T,T,device=x.device,dtype=torch.bool)
            for i in range(T):
                start=max(0,i-self.sparse_w);allow[i,start:i+1]=True
                allow[i,idx[i]]=True
            mask=torch.where(~allow,float('-inf'),0.0)
        else:
            mask=torch.triu(torch.full((T,T),float('-inf'),device=x.device),diagonal=1)
        a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
        a=a+mask.unsqueeze(0).unsqueeze(0);a=F.softmax(a,-1);a=self.attn_drop(a);y=a@v
        y=y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.c_proj(y)), cq

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.n_embd*4*2//3//256*256
        self.w1=nn.Linear(c.n_embd,h,bias=c.bias);self.w2=nn.Linear(h,c.n_embd,bias=c.bias);self.w3=nn.Linear(c.n_embd,h,bias=c.bias)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))

class Block(nn.Module):
    def __init__(self,c,layer_idx):
        super().__init__()
        self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.mlp=SwiGLU(c)
        self.is_attn=layer_idx in c.attention_layers or layer_idx in c.sparse_layers
        if layer_idx in c.inertia_layers:
            self.layer=InertiaLayer(c)
        else:
            self.layer=MLALayer(c,sparse=layer_idx in c.sparse_layers)
    def forward(self,x,latents=None):
        if self.is_attn:
            a,cq=self.layer(self.ln1(x),latents)
            x=x+a
        else:
            x=x+self.layer(self.ln1(x))
        return x+self.mlp(self.ln2(x)), cq if self.is_attn else None

class RINA_AC(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c,i) for i in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd,bias=c.bias),
        ))
        self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=c.bias)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._iw)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p,0,0.02/math.sqrt(2*c.n_layer))
        core=sum(p.numel() for p in self.parameters())-self.transformer.wte.weight.numel()
        print(f'RINA_AC: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')
    def _iw(self,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx,targets=None):
        B,T=idx.size();assert T<=self.config.block_size
        x=self.transformer.drop(self.transformer.wte(idx))
        for b in self.transformer.h:
            x,_=b(x)
        x=self.transformer.ln_f(x)
        if targets is not None:
            l=self.lm_head(x)
            loss=F.cross_entropy(l.view(-1,l.size(-1)),targets.reshape(-1),ignore_index=-1)
            return l,loss
        return self.lm_head(x[:,[-1],:]),None
