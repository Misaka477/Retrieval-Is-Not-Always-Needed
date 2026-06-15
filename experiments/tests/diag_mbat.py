"""诊断：untrained MBAT 的 CE 和 logits 分布"""
import os, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DM=384; V=65536; NL=8; NH=6; HD=DM//NH; KV=HD//2; SEQ=512

class A(nn.Module):
    def __init__(s,mk=64,lt='cbtka'):
        super().__init__();s.mk=mk;s.lt=lt
        s.q=nn.Linear(DM,DM);s.qn=nn.LayerNorm(DM);s.kp=nn.Linear(DM,KV*NH);s.kn=nn.LayerNorm(KV*NH);s.ke=nn.Linear(KV*NH,DM)
        s.vp=nn.Linear(DM,KV*NH);s.ve=nn.Linear(KV*NH,DM);s.o=nn.Linear(DM,DM);s.register_buffer('pb',torch.zeros(1,SEQ,DM))
        p=torch.arange(SEQ).unsqueeze(1);d=torch.arange(HD//2).unsqueeze(0);pe=torch.zeros(1,SEQ,HD)
        pe[0,:,0::2]=torch.sin(p/10000**(2*d/HD));pe[0,:,1::2]=torch.cos(p/10000**(2*d/HD));s.pb[:,:,:HD]=pe
    def forward(s,x):
        B,T,D=x.shape;k=min(s.mk,T)
        q=s.qn(s.q(x)).view(B,T,NH,HD);kl=s.kn(s.kp(x));ke=s.ke(kl).view(B,T,NH,HD);v=s.vp(x).view(B,T,NH,KV)
        pe=s.pb[:,:T,:HD];q=(q+pe.view(1,T,1,HD)).transpose(1,2);ke=(ke+pe.view(1,T,1,HD)).transpose(1,2);v=v.transpose(1,2)
        sc=torch.matmul(q,ke.transpose(-2,-1))/math.sqrt(HD)
        if s.lt=='window':
            W=s.mk//2;m=torch.zeros_like(sc)
            for i in range(T):S=max(0,i-W);E=min(T,i+W+1);m[:,:,i,S:E]=1
            at=F.softmax(sc.masked_fill(m==0,float('-inf')),dim=-1)
        else:
            _,idx=torch.topk(sc,k,dim=-1);m=torch.zeros_like(sc).scatter_(-1,idx,1.0)
            at=F.softmax(sc.masked_fill(m==0,float('-inf')),dim=-1)
        h=torch.matmul(at,v).transpose(1,2).contiguous().view(B,T,-1);return s.o(s.ve(h))
class B(nn.Module):
    def __init__(s,i):
        super().__init__();s.ln1=nn.LayerNorm(DM);s.ln2=nn.LayerNorm(DM)
        mk=32 if i<3 else(64 if i<6 else 96);lt='window'if i<3 else'cbtka'
        s.attn=A(mk,lt);s.ffn=nn.Sequential(nn.Linear(DM,DM*4),nn.GELU(),nn.Linear(DM*4,DM))
    def forward(s,x):return x+s.attn(s.ln1(x)),x+s.ffn(s.ln2(x))
class M(nn.Module):
    def __init__(s):
        super().__init__();s.emb=nn.Embedding(V,DM);s.blocks=nn.ModuleList([B(i)for i in range(NL)]);s.ln=nn.LayerNorm(DM);s.head=nn.Linear(DM,V,bias=False)
        s.apply(s._init)
    def _init(s,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0,0.02)
        elif isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0,0.02)
    def forward(s,x):
        h=s.emb(x)
        for blk in s.blocks:h,_=blk(h)
        return s.head(s.ln(h))

m=M().to('cuda')
data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r')
N=data.shape[0];VS=N-N//10;rng=np.random.RandomState(42)
pos=rng.randint(VS,N-513,4)
x=torch.stack([torch.from_numpy(data[p:p+512].copy()).long()for p in pos]).to('cuda')
with torch.no_grad():
    l=m(x);ce=F.cross_entropy(l[:,:-1].reshape(-1,V),x[:,1:].reshape(-1));print(f'CE={ce.item():.4f}')
    print(f'Logits:[{l.min():.1f},{l.max():.1f}]');probs=F.softmax(l,dim=-1);print(f'Max prob={probs.max():.6f}')
    # No attention: head(ln(emb(x)))
    h=m.emb(x);l2=m.head(m.ln(h));ce2=F.cross_entropy(l2[:,:-1].reshape(-1,V),x[:,1:].reshape(-1));print(f'CE(no_attn)={ce2.item():.4f}')
    cos=F.cosine_similarity(m.head.weight,m.emb.weight).mean();print(f'head-emb cos={cos:.4f}')
    # head(ln(emb(x))) reconstruct x
    l3=m.head(m.ln(m.emb(x)));r=F.cross_entropy(l3.reshape(-1,V),x.reshape(-1));print(f'CE(recon x)={r.item():.4f}')
