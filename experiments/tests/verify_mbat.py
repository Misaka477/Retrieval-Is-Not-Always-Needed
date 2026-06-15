"""一步验证：MBAT 初始化正确、FFN 生效、验证隔离"""
import os, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device='cuda';V=65536;DM=384;NL=8;NH=6;HD=DM//NH;KV=HD//2;SEQ=512

class A(nn.Module):
    def __init__(self,mk=64,lt='cbtka'):
        super().__init__();self.mk=mk;self.lt=lt
        self.q=nn.Linear(DM,DM);self.qn=nn.LayerNorm(DM);self.kp=nn.Linear(DM,KV*NH);self.kn=nn.LayerNorm(KV*NH)
        self.ke=nn.Linear(KV*NH,DM);self.vp=nn.Linear(DM,KV*NH);self.ve=nn.Linear(KV*NH,DM);self.o=nn.Linear(DM,DM)
        self.register_buffer('pb',torch.zeros(1,SEQ,DM))
        p=torch.arange(SEQ).unsqueeze(1);d=torch.arange(HD//2).unsqueeze(0);pe=torch.zeros(1,SEQ,HD)
        pe[:,:,0::2]=torch.sin(p/10000**(2*d/HD))[None];pe[:,:,1::2]=torch.cos(p/10000**(2*d/HD))[None]
        self.pb[:,:,:HD]=pe
    def forward(self,x):
        B,T,D=x.shape;k=min(self.mk,T)
        q=self.qn(self.q(x)).view(B,T,NH,HD);kl=self.kn(self.kp(x));ke=self.ke(kl).view(B,T,NH,HD)
        v=self.vp(x).view(B,T,NH,KV);pe=self.pb[:,:T,:HD]
        q=(q+pe.view(1,T,1,HD)).transpose(1,2);ke=(ke+pe.view(1,T,1,HD)).transpose(1,2);v=v.transpose(1,2)
        sc=torch.matmul(q,ke.transpose(-2,-1))/math.sqrt(HD)
        if self.lt=='window':
            W=self.mk//2;m_=torch.zeros_like(sc)
            for i in range(T):S=max(0,i-W);E=min(T,i+W+1);m_[:,:,i,S:E]=1
            at=F.softmax(sc.masked_fill(m_==0,float('-inf')),dim=-1)
        else:
            _,idx=torch.topk(sc,k,dim=-1);m_=torch.zeros_like(sc).scatter_(-1,idx,1.0)
            at=F.softmax(sc.masked_fill(m_==0,float('-inf')),dim=-1)
        h=torch.matmul(at,v).transpose(1,2).contiguous().view(B,T,-1)
        return self.o(self.ve(h))

class B(nn.Module):
    def __init__(self,i):
        super().__init__();self.ln1=nn.LayerNorm(DM);self.ln2=nn.LayerNorm(DM)
        mk=32 if i<3 else(64 if i<6 else 96);lt='window'if i<3 else'cbtka'
        self.attn=A(mk,lt);self.ffn=nn.Sequential(nn.Linear(DM,DM*4),nn.GELU(),nn.Linear(DM*4,DM))
    def forward(self,x):x=x+self.attn(self.ln1(x));x=x+self.ffn(self.ln2(x));return x

class M(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(V,DM);self.blocks=nn.ModuleList([B(i)for i in range(NL)])
        self.ln=nn.LayerNorm(DM);self.head=nn.Linear(DM,V,bias=False)
        self.apply(self._init)
    def _init(self,m):isinstance(m,nn.Linear)and nn.init.normal_(m.weight,0,0.02);isinstance(m,nn.Embedding)and nn.init.normal_(m.weight,0,0.02)
    def forward(self,x):
        h=self.emb(x)
        for b in self.blocks:h=b(h)
        return self.head(self.ln(h))

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r')
N=data.shape[0];TE=N-N//10
m=M().to(device);print(f'Params:{sum(p.numel()for p in m.parameters())/1e6:.2f}M')
opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
pos=np.random.randint(0,TE-SEQ-1,8)
x=torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long()for p in pos]).to(device)
loss=F.cross_entropy(m(x)[:,:-1].reshape(-1,V),x[:,1:].reshape(-1))
print(f'Train CE:{loss.item():.2f}')
vrng=np.random.RandomState(42)
p_=vrng.randint(TE,N-SEQ-1,8)
vx=torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long()for p in p_]).to(device)
ce_v=F.cross_entropy(m(vx)[:,:-1].reshape(-1,V),vx[:,1:].reshape(-1)).item()
print(f'Val CE:{ce_v:.2f}')
opt.zero_grad();loss.backward()
gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
opt.step();print(f'Grad norm:{gn:.2f}')
print('OK')
