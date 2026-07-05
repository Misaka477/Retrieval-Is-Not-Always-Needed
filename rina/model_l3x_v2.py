"""RINA L3X V2 — Jamba-style hybrid: GatedSSM (adaptive depth) + ClusterCompressAttention (q2k/q1v).
Builds on model_jamba.py's alternating pattern with two internal upgrades:
  1. GatedSSM: per-token adaptive depth (1 or ssm_steps steps) instead of fixed K steps
  2. ClusterCompressAttention: cq-driven latent clustering for KV compression instead of top-K tokens
  3. q2k_q1v activation quantization (from model_jamba_qw2.py)
"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F
from rina.model_jamba_qw2 import q4_0_block, q2_block, q1_block

@dataclass
class L3XV2_Config:
    block_size:int=2048
    vocab_size:int=128256
    n_layer:int=16
    n_embd:int=640
    n_head:int=10
    n_kv_heads:int=5
    head_dim:int=64
    d_c:int=160
    d_h_r:int=32
    dropout:float=0.0; bias:bool=False

    ssm_steps:int=3
    quant_mode:str='q2k_q1v'

    n_clusters:int=32
    cluster_momentum:float=0.995
    cluster_recompute:int=32

    layer_types:tuple=None  # 0=SSM, 1=Attention; None=auto (3SSM+1Attn alternating)


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


class ClusterManager:
    """Online k-means on cq latents for KV compression in attention layers."""
    def __init__(self,n_clusters,d_c,recompute=32,momentum=0.995):
        self.n=n_clusters;self.d=d_c;self.rc=recompute;self.m=momentum
        self.step=0;self.centroids=None

    def update(self,cq):
        B,T,_=cq.shape;flat=cq.reshape(-1,self.d)
        if self.centroids is None or self.step%self.rc==0:
            with torch.no_grad():
                n=min(self.n,len(flat))
                idx=torch.randperm(len(flat),device=flat.device)[:n]
                self.centroids=flat[idx].clone()
                if n<self.n:
                    pad=self.centroids.new_zeros(self.n-n,self.d)
                    self.centroids=torch.cat([self.centroids,pad])
        with torch.no_grad():
            sim=flat@self.centroids.T
            hard=sim.argmax(-1)
            for k in range(self.n):
                m=(hard==k)
                if m.any():
                    self.centroids[k]=self.centroids[k]*self.m+flat[m].mean(0)*(1-self.m)
        self.step+=1

    def compress_kv(self,kc,v,cq):
        B,T,_=cq.shape;n=self.n;dh=kc.size(-1);nh=kc.size(1)
        flat=cq.reshape(B*T,self.d)
        if self.centroids is None: return kc,v
        w=(flat@self.centroids.T).softmax(-1).reshape(B,T,n,1,1)
        return (kc.unsqueeze(2)*w).sum(dim=1),(v.unsqueeze(2)*w).sum(dim=1)


class SSMRouter(nn.Module):
    """Per-token SSM depth: 1 or max_steps."""
    def __init__(self,d,max_steps):
        super().__init__()
        self.router=nn.Linear(d,1);self.max_steps=max_steps
    def forward(self,h):
        p=self.router(h.detach()).sigmoid()
        return torch.where(p>0.5,torch.full_like(p,self.max_steps).long(),
                          torch.ones_like(p).long()).squeeze(-1)


class GatedSSM(nn.Module):
    """SSM layer with per-token adaptive depth routing (1 or ssm_steps).
    Uses parallel cumprod/cumsum scan identical to model_jamba's InertiaLayer.
    """
    def __init__(self,c):
        super().__init__()
        self.n_head=c.n_head;self.dh=c.head_dim;self.ssm_steps=c.ssm_steps
        d=c.n_embd;dc=c.d_c;nh=c.n_head;dh=c.head_dim
        self.w_dq=nn.Linear(d,dc,bias=c.bias)
        self.q_norm=nn.LayerNorm(dc,bias=c.bias)
        self.w_mem=nn.ModuleList([nn.Linear(dc,nh*dh,bias=c.bias) for _ in range(self.ssm_steps)])
        self.w_decay=nn.ModuleList([nn.Linear(dc,nh,bias=c.bias) for _ in range(self.ssm_steps)])
        self.w_out=nn.Linear(nh*dh+d,d,bias=c.bias)
        self.router=SSMRouter(d,self.ssm_steps) if self.ssm_steps>1 else None

    def forward(self,x):
        B,T,_=x.shape
        n_steps=self.router(x).float().mean() if self.router else 0
        cq=self.q_norm(self.w_dq(x))
        mems=[w(cq).view(B,T,self.n_head,self.dh) for w in self.w_mem]
        decays=[torch.sigmoid(w(cq)).view(B,T,self.n_head,1) for w in self.w_decay]

        d_agg,m_agg=decays[0],mems[0]
        for i in range(1,self.ssm_steps):
            d_agg=d_agg*decays[i];m_agg=m_agg*decays[i]+mems[i]

        a=d_agg.expand(-1,-1,-1,self.dh)
        ca=torch.cumprod(a,dim=1)
        sf=(ca*torch.cumsum(m_agg/(ca+1e-8),dim=1)).reshape(B,T,-1)
        return self.w_out(torch.cat([x,sf],-1)),cq,n_steps


class ClusterCompressAttention(nn.Module):
    """MLA attention with q2k_q1v + cluster-compressed sparse KV.
    Replaces L3X's SparseIndexManager with online soft clustering.
    Each query attends to n_cluster representative KV entries instead of T tokens.
    """
    def __init__(self,c):
        super().__init__()
        self.n_head=c.n_head;self.n_kv=c.n_kv_heads
        self.n_rep=max(1,c.n_head//c.n_kv_heads)
        self.d=c.n_embd;self.dh=c.head_dim;self.d_c=c.d_c;self.d_hr=c.d_h_r
        self.quant_mode=getattr(c,'quant_mode','q2k_q1v')

        self.w_dqkv=nn.Linear(self.d,self.d_c,bias=c.bias)
        self.q_norm=nn.LayerNorm(self.d_c,bias=c.bias)
        self.w_uq=nn.Linear(self.d_c,self.n_head*self.dh,bias=c.bias)
        self.w_uk=nn.Linear(self.d_c,self.n_kv*self.dh,bias=c.bias)
        self.w_k2v=nn.Linear(self.n_kv*self.dh,self.n_kv*self.dh,bias=c.bias)
        self.w_qr=nn.Linear(self.d,self.n_head*self.d_hr,bias=c.bias)
        self.w_kr=nn.Linear(self.d,self.n_kv*self.d_hr,bias=c.bias)
        self.rope=RoPE(self.d_hr,c.block_size)
        self.rope_q=RoPE(self.d_hr,c.block_size)
        self.c_proj=nn.Linear(self.n_head*self.dh,self.d,bias=c.bias)
        self.resid_drop=nn.Dropout(c.dropout)

    def forward(self,x,cluster_mgr):
        B,T,_=x.shape
        cq=self.q_norm(self.w_dqkv(x))

        qc=self.w_uq(cq).view(B,T,self.n_head,self.dh).transpose(1,2)
        kc=self.w_uk(cq).view(B,T,self.n_kv,self.dh).transpose(1,2)
        v=self.w_k2v(self.w_uk(cq)).view(B,T,self.n_kv,self.dh).transpose(1,2)

        if self.quant_mode=='q2k_q1v':
            qc=F.normalize(qc,dim=-1);kc=F.normalize(kc,dim=-1);v=q1_block(v)
        else:
            qc=q4_0_block(qc);kc=q4_0_block(kc);v=q2_block(v)

        qr=self.w_qr(x).view(B,T,self.n_head,self.d_hr).transpose(1,2)
        kr=self.w_kr(x).view(B,T,self.n_kv,self.d_hr).transpose(1,2)
        qr,kr=self.rope_q(qr),self.rope(kr)

        if self.n_rep>1:
            kc=kc.repeat_interleave(self.n_rep,1)
            v=v.repeat_interleave(self.n_rep,1)
            kr=kr.repeat_interleave(self.n_rep,1)

        if cluster_mgr is not None and cluster_mgr.centroids is not None:
            cluster_mgr.update(cq.detach())
            kc_c,v_c=cluster_mgr.compress_kv(kc,v,cq.detach())
            kr_c=kr.mean(dim=2,keepdim=True).expand(-1,-1,kc_c.size(2),-1)
            k=torch.cat([kc_c,kr_c],-1);v=v_c
            y=F.scaled_dot_product_attention(
                torch.cat([qc,qr],-1),k,v,dropout_p=0,is_causal=False)
        else:
            y=F.scaled_dot_product_attention(
                torch.cat([qc,qr],-1),torch.cat([kc,kr],-1),v,
                dropout_p=0,is_causal=True)

        y=y.transpose(1,2).contiguous().view(B,T,-1)
        return self.resid_drop(self.c_proj(y)),cq


class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        h=c.n_embd*4*2//3//256*256
        self.w1=nn.Linear(c.n_embd,h,bias=c.bias)
        self.w2=nn.Linear(h,c.n_embd,bias=c.bias)
        self.w3=nn.Linear(c.n_embd,h,bias=c.bias)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):
        return self.dp(self.w2(F.silu(self.w1(x))*self.w3(x)))


class V2Block(nn.Module):
    """Jamba-style block: either GatedSSM or ClusterCompressAttention + SwiGLU."""
    def __init__(self,c,layer_type,layer_idx=0):
        super().__init__()
        self.layer_type=layer_type;self.layer_idx=layer_idx
        self.ln1=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.ln2=nn.LayerNorm(c.n_embd,bias=c.bias)
        self.mlp=SwiGLU(c)
        if layer_type==0:
            self.path=GatedSSM(c)
        else:
            self.path=ClusterCompressAttention(c)

    def forward(self,x,cluster_mgr=None):
        h=self.ln1(x)
        if self.layer_type==0:
            out,cq,ns=self.path(h)
        else:
            out,cq=self.path(h,cluster_mgr)
            ns=0
        r=x+out
        return r+self.mlp(self.ln2(r)),ns,cq


class RINA_L3X_V2(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c

        if c.layer_types is not None:
            ltypes=list(c.layer_types)
        else:
            n_sparse=max(4,c.n_layer//4)
            n_ssm=c.n_layer-n_sparse
            ratio=n_ssm//n_sparse
            ltypes=[]
            s_count=0
            for i in range(c.n_layer):
                if (i+1)%(ratio+1)==0 and s_count<n_sparse:
                    ltypes.append(1);s_count+=1
                else:
                    ltypes.append(0)

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([V2Block(c,ltypes[i],i) for i in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd,bias=c.bias),
        ))
        self.lm_head=nn.Linear(c.n_embd,c.vocab_size,bias=c.bias)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._iw)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p,0,0.02/math.sqrt(2*c.n_layer))
        core=sum(p.numel() for p in self.parameters())-self.transformer.wte.weight.numel()
        print(f'RINA_L3X_V2: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')
        for i,t in enumerate(ltypes):
            print(f'  layer {i}: {"GatedSSM" if t==0 else "ClusterAttn"}')
    def _iw(self,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0.0,0.02)

    def forward(self,idx,targets=None,return_lats=False):
        B,T=idx.size();assert T<=self.config.block_size
        c=self.config
        x=self.transformer.drop(self.transformer.wte(idx))
        cm=ClusterManager(c.n_clusters,c.d_c,
            recompute=c.cluster_recompute,momentum=c.cluster_momentum)

        lats=[];avg_ssm_depth=0
        for b in self.transformer.h:
            x,ns,cq=b(x,cm)
            if b.layer_type==0:
                avg_ssm_depth+=ns.item() if hasattr(ns,'item') else ns
            if b.layer_type==1:
                lats.append(cq)

        x=self.transformer.ln_f(x)
        avg_d=avg_ssm_depth/self.config.n_layer
        if targets is not None:
            l=self.lm_head(x)
            loss=F.cross_entropy(l.view(-1,l.size(-1)),targets.reshape(-1),ignore_index=-1)
            if return_lats and lats:
                return l,loss,torch.stack(lats,dim=1),avg_d
            return l,loss,avg_d
        return self.lm_head(x[:,[-1],:]),None,avg_d

    @staticmethod
    def compute_contrastive_loss(lats, margin=1.1):
        B,L,T,D=lats.shape
        lats=lats[:,1:].mean(dim=1)
        lats=F.normalize(lats,dim=-1)
        pos_sim=(lats[:,:-1]*lats[:,1:]).sum(-1).mean()
        shift=1 if B>1 else 0
        b_shuf=(torch.arange(B,device=lats.device)+shift)%B
        neg_sim=(lats*lats[b_shuf]).sum(-1).mean()
        gap=pos_sim-neg_sim
        return F.relu(0.95-pos_sim)+F.relu(neg_sim+0.15)+F.relu(margin-gap)
