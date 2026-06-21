"""RINA CF — Confidence-based routing: 置信度决定路径, 越低越用力"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F

def ste_round(x): return x + (x.round()-x).detach()
def q4(x,gs=32):
    o=x.shape;xf=x.reshape(-1,gs);s=xf.abs().max(-1,keepdim=True).values/7.0
    xq=ste_round(xf/(s+1e-8)).clamp(-7,7);return (xq*s).reshape(o)
def q2(x,gs=32):
    o=x.shape;xf=x.reshape(-1,gs);s=xf.abs().max(-1,keepdim=True).values/1.0
    xq=ste_round(xf/(s+1e-8)).clamp(-1,1);return (xq*s).reshape(o)

@dataclass
class RINA_CF_Config:
    block_size:int=1024;vocab_size:int=50304;n_layer:int=12;n_embd:int=512
    n_head:int=8;n_kv_heads:int=4;head_dim:int=64;d_c:int=128;d_h_r:int=32
    dropout:float=0.0;bias:bool=False;use_int4:bool=False
    sparse_k:int=8;sparse_local_w:int=4
    confidence_thresholds:tuple=(3.0,6.0,8.0)  # 原始熵阈值: L1/L3/L2/ROM

class RoPE(nn.Module):
    def __init__(self,dim,max_len=4096,base=10000.0):
        super().__init__();self.dim=dim
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
    def forward(self,x,rom_kv=None):
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
        if rom_kv is not None:
            k_rom,v_rom=rom_kv
            kr_rom=torch.zeros(B,self.n_head,k_rom.size(2),self.d_hr,device=x.device)
            kc=torch.cat([k_rom,kc],dim=2);v=torch.cat([v_rom,v],dim=2);kr=torch.cat([kr_rom,kr],dim=2)
        nk=kc.size(2)
        q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1)
        a=(q@k.transpose(-2,-1))/math.sqrt(q.size(-1))
        mask=torch.triu(torch.full((T,nk),float('-inf'),device=x.device),diagonal=1)
        a=a+mask.unsqueeze(0).unsqueeze(0);a=F.softmax(a,-1);a=self.attn_drop(a);y=a@v
        y=y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.c_proj(y)), cq

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__();h=c.n_embd*4*2//3//256*256
        self.w1=nn.Linear(c.n_embd,h,bias=c.bias);self.w2=nn.Linear(h,c.n_embd,bias=c.bias);self.w3=nn.Linear(c.n_embd,h,bias=c.bias)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))

class CFBlock(nn.Module):
    """置信度调度块: 训练时全量 attention, 推理时按置信度选路"""
    def __init__(self,c):
        super().__init__()
        self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.mlp=SwiGLU(c)
        self.l1=InertiaLayer(c)
        self.l3=MLALayer(c,sparse=True)
        self.l2=MLALayer(c,sparse=False)
        self.conf_head=nn.Linear(c.d_c,1)
        self.th=c.confidence_thresholds
    def forward(self,x,rom_kv=None,return_conf=False):
        ln=self.ln1(x)
        cq=self.l2.q_norm(self.l2.w_dqkv(ln))
        raw=self.conf_head(cq)
        conf=raw  # 线性输出, 直接逼近原始熵值
        
        if return_conf:
            avg=conf.mean().item()
            if avg<self.th[0]:
                out=self.l1(ln)
            elif avg<self.th[1]:
                out,_=self.l3(ln)
            elif avg<self.th[2]:
                out,_=self.l2(ln)
            else:
                out,_=self.l2(ln,rom_kv)
            return x+out+self.mlp(self.ln2(x)),conf
        out,_=self.l2(ln)
        return x+out+self.mlp(self.ln2(x)),conf

class RINA_CF(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([CFBlock(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd,bias=c.bias),
        ))
        self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=c.bias)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._iw)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p,0,0.02/math.sqrt(2*c.n_layer))
        for n,m in self.named_modules():
            if isinstance(m,nn.Linear) and 'conf_head' in n:
                nn.init.xavier_uniform_(m.weight,0.01)
        core=sum(p.numel() for p in self.parameters())-self.transformer.wte.weight.numel()
        print(f'RINA_CF: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')
    def _iw(self,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx,targets=None,rom_kv=None,return_conf=False):
        B,T=idx.size();assert T<=self.config.block_size
        x=self.transformer.drop(self.transformer.wte(idx))
        confs=[]
        for b in self.transformer.h:
            x,c=b(x,rom_kv,return_conf);confs.append(c)
        x=self.transformer.ln_f(x)
        confs=torch.cat(confs,dim=-1)  # [B,T,L]
        if targets is not None:
            l=self.lm_head(x)
            ce=F.cross_entropy(l.view(-1,l.size(-1)),targets.reshape(-1),ignore_index=-1)
            with torch.no_grad():
                p=F.softmax(l,dim=-1)
                real_ent=-(p*p.clamp(1e-8).log()).sum(-1,keepdim=True)  # 原始熵, 不归一化
            return l,ce,F.mse_loss(confs.mean(-1,keepdim=True),real_ent),confs
        return self.lm_head(x[:,[-1],:]),None,None,None
