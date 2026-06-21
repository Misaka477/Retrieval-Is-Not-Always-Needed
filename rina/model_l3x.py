"""RINA L3X — Enhanced Sparse Attention with Sliding Window + Index Sharing"""
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
class RINA_L3X_Config:
    block_size:int=1024;vocab_size:int=50304;n_layer:int=12;n_embd:int=512
    n_head:int=8;n_kv_heads:int=4;head_dim:int=64;d_c:int=128;d_h_r:int=32
    dropout:float=0.0;bias:bool=False;use_int4:bool=False
    sparse_k:int=8;sparse_local_w:int=4
    sparse_window:int=32  # 滑窗大小 W，W 步重算一次索引

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

class SparseIndexManager:
    """滑窗索引管理器：跨步复用 + 层间共享"""
    def __init__(self, window=32, k=8, local_w=4):
        self.window=window;self.k=k;self.local_w=local_w
        self.reset()
    def reset(self):
        self.step=0
        self.mask=None  # [T,T] bool, True=allow attend
    def should_recompute(self):
        return self.mask is None or self.step%self.window==0
    def recompute(self, cq, T, device):
        """全量计算 top-K 索引（取第一个 batch item）"""
        sim=torch.bmm(F.normalize(cq[:1],-1),F.normalize(cq[:1],-1).transpose(1,2))
        _,idx=torch.topk(sim.squeeze(0),min(self.k,T),dim=-1)
        allow=torch.zeros(T,T,device=device,dtype=torch.bool)
        for i in range(T):
            start=max(0,i-self.local_w);allow[i,start:i+1]=True
            allow[i,idx[i]]=True
        self.mask=allow
    def extend(self, T, device):
        """扩展 mask 到新长度（新 token 行+列追加）"""
        old_T=self.mask.size(0)
        if T<=old_T: return
        new=torch.zeros(T,T,device=device,dtype=torch.bool)
        new[:old_T,:old_T]=self.mask
        # 新 token attend 自己和最近 local_w 个历史 token
        for i in range(old_T,T):
            start=max(0,i-self.local_w);new[i,start:i+1]=True
            new[i,i]=True
        self.mask=new
    def step_forward(self, cq, T, device):
        """每生成一步调用一次，计算/刷新 mask"""
        if self.should_recompute():
            self.recompute(cq,T,device)
        elif T>self.mask.size(0):
            self.extend(T,device)
        self.step+=1
        return self.mask  # 返回 bool mask
    def get_mask(self, T):
        """attention 内部调用来获取当前 mask（不触发重算）"""
        return torch.where(self.mask,0.0,float('-inf'))

class MLALayer(nn.Module):
    """MLA 注意力 + 滑窗稀疏索引支持"""
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
    def forward(self,x,index_mgr=None):
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
        if self.sparse and index_mgr is not None and T>1:
            mask=index_mgr.get_mask(T)
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
    def __init__(self,c,is_sparse=False):
        super().__init__()
        self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.attn=MLALayer(c,sparse=is_sparse)
        self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.mlp=SwiGLU(c)
    def forward(self,x,index_mgr=None):
        a,_=self.attn(self.ln1(x),index_mgr)
        return x+a+self.mlp(self.ln2(x))

class RINA_L3X(nn.Module):
    """模型：全量 attention 训练，稀疏索引推理"""
    def __init__(self,c):
        super().__init__();self.config=c
        # 前 8 层全量 attention，后 4 层稀疏 attention
        n_sparse=4
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c,is_sparse=i>=c.n_layer-n_sparse) for i in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd,bias=c.bias),
        ))
        self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=c.bias)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._iw)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p,0,0.02/math.sqrt(2*c.n_layer))
        core=sum(p.numel() for p in self.parameters())-self.transformer.wte.weight.numel()
        print(f'RINA_L3X: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')
    def _iw(self,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx,targets=None,use_sparse=False):
        B,T=idx.size();assert T<=self.config.block_size
        x=self.transformer.drop(self.transformer.wte(idx))
        mgr=SparseIndexManager(window=self.config.sparse_window,k=self.config.sparse_k,
                              local_w=self.config.sparse_local_w) if use_sparse else None
        for b in self.transformer.h:
            x=b(x,mgr if use_sparse else None)
        x=self.transformer.ln_f(x)
        if targets is not None:
            l=self.lm_head(x)
            loss=F.cross_entropy(l.view(-1,l.size(-1)),targets.reshape(-1),ignore_index=-1)
            return l,loss
        return self.lm_head(x[:,[-1],:]),None
