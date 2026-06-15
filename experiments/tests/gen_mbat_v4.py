"""生成测试 — 从训练脚本只拷贝模型定义，不触发训练"""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, ast
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device='cuda';V=65536;DM=384;NL=8;NH=6;NG=2;HD=DM//NH;DC_G=HD*4//NG;DC_Q=HD*4;HR=HD//2;SEQ=128

# 从 train_mbat_v4_local.py 拷过来的模型定义 (完全一致)
class BL(nn.Module):
    def __init__(s,i,o,b=False):
        super().__init__();s.w=nn.Parameter(torch.empty(o,i));s.b=nn.Parameter(torch.zeros(o))if b else None
        nn.init.normal_(s.w,0,0.02);s.g=max(1,o//128);s.s=nn.Parameter(torch.ones(s.g))
    def forward(s,x):
        w=s.w;g=s.g;wg=w.view(g,-1);sc=wg.abs().mean(-1,keepdim=True)*s.s.view(g,1)
        wq=torch.clamp(torch.round(wg/(sc+1e-8)),-1,1)*sc;wq=wq.view_as(w)
        return F.linear(x,w+(wq-w).detach(),s.b)

def q8(x):
    s=x.abs().max(dim=-1,keepdim=True).values/127
    return x+(torch.clamp(torch.round(x/(s+1e-8)),-127,127)*s-x).detach()

class PA(nn.Module):
    def __init__(s,mk=64,lt='cbtka'):
        super().__init__();s.mk=mk;s.lt=lt;s.G=NG
        s.WDKV=nn.ModuleList([BL(DM,DC_G)for _ in range(s.G)])
        s.KVn=nn.ModuleList([nn.LayerNorm(DC_G)for _ in range(s.G)])
        s.WUK=nn.ModuleList([BL(DC_G,DM//s.G)for _ in range(s.G)])
        s.WUV=nn.ModuleList([BL(DC_G,DM//s.G)for _ in range(s.G)])
        s.WDQ=BL(DM,DC_Q);s.QN=nn.LayerNorm(DC_Q);s.WUQ=BL(DC_Q,DM)
        s.WQR=BL(DC_Q,HR*NH);s.WKR=BL(DM,HR);s.WO=BL(DM,DM)
        s.register_buffer('c',torch.zeros(1,SEQ,HR));s.register_buffer('si',torch.zeros(1,SEQ,HR))
        inv=1.0/(10000**(torch.arange(0,HR,2).float()/HR));p=torch.arange(SEQ).float()
        s.c[0,:,0::2]=torch.cos(p[:,None]*inv[None,:]);s.c[0,:,1::2]=s.c[0,:,0::2]
        s.si[0,:,0::2]=torch.sin(p[:,None]*inv[None,:]);s.si[0,:,1::2]=s.si[0,:,0::2]
    def ar(s,x,p):c=s.c[:,p,:].unsqueeze(2);i=s.si[:,p,:].unsqueeze(2);xr=torch.cat([-x[...,1::2],x[...,0::2]],-1);return x*c+xr*i
    def ca(s,qt,kt,st,mk,ca_,T):
        B,H,_,D=qt.shape;qc=qt[:,:,::st,:];kc=kt[:,:,::st,:];Tc=qc.size(2)
        sc=torch.matmul(qc,kc.transpose(-2,-1))/math.sqrt(D);sc=sc+ca_[::st,::st].unsqueeze(0).unsqueeze(0)
        sf=torch.zeros(B,H,T,T,device=qt.device);sf[:,:,::st,::st]=sc
        return F.interpolate(sf.view(B*H,1,T,T),size=(T,T),mode='bilinear',align_corners=False).view(B,H,T,T).masked_fill(ca_==float('-inf'),float('-inf'))
    def forward(s,x):
        B,T,D=x.shape;kt=min(s.mk,T);x=q8(x)
        kcl,vcl=[],[]
        for g in range(s.G):
            ck=s.KVn[g](s.WDKV[g](x))
            kcl.append(s.WUK[g](ck).view(B,T,NH//s.G,HD))
            vcl.append(s.WUV[g](ck).view(B,T,NH//s.G,HD))
        kc=torch.cat(kcl,2);vc=torch.cat(vcl,2)
        cq=s.QN(s.WDQ(x));qc=s.WUQ(cq).view(B,T,NH,HD)
        qr=s.WQR(cq).view(B,T,NH,HR);kr=s.WKR(x).unsqueeze(2).expand(-1,-1,NH,-1)
        qr=s.ar(qr,torch.arange(T,device=x.device));kr=s.ar(kr,torch.arange(T,device=x.device))
        q=torch.cat([qc,qr],-1);k_=torch.cat([kc,kr],-1);qt=q.transpose(1,2);kt_=k_.transpose(1,2);vt=vc.transpose(1,2)
        ca_=torch.triu(torch.full((T,T),float('-inf'),device=x.device),diagonal=1)
        if s.lt=='window':
            sc=torch.matmul(qt,kt_.transpose(-2,-1))/math.sqrt(HD+HR)+ca_;W=s.mk//2;m=torch.zeros_like(sc)
            for i in range(T):S=max(0,i-W);E=min(T,i+W+1);m[:,:,i,S:E]=1
            at=F.softmax(sc.masked_fill(m==0,float('-inf')),-1)
            h=torch.matmul(at,vt).transpose(1,2).contiguous().view(B,T,-1);return s.WO(h)
        else:
            if T>256:
                sc_l1=s.ca(qt,kt_,4,kt*2,ca_,T);_,idx=torch.topk(sc_l1,kt,-1)
            else:
                sc=torch.matmul(qt,kt_.transpose(-2,-1))/math.sqrt(HD+HR)+ca_;_,idx=torch.topk(sc,kt,-1)
            ik=idx.unsqueeze(-1).expand(-1,-1,-1,-1,HD+HR);iv=idx.unsqueeze(-1).expand(-1,-1,-1,-1,HD)
            ktk=torch.gather(kt_.unsqueeze(3).expand(-1,-1,-1,kt,-1),2,ik)
            sc=(qt.unsqueeze(3)*ktk).sum(-1)/math.sqrt(HD+HR);at=F.softmax(sc,-1)
            vtk=torch.gather(vt.unsqueeze(3).expand(-1,-1,-1,kt,-1),2,iv)
            return s.WO((at.unsqueeze(-1)*vtk).sum(3).transpose(1,2).contiguous().view(B,T,-1))

class BLK(nn.Module):
    def __init__(s,i):
        super().__init__();s.ln1=nn.LayerNorm(DM);s.ln2=nn.LayerNorm(DM)
        mk=32 if i<3 else(64 if i<6 else 96);lt='window'if i<3 else'cbtka'
        s.attn=PA(mk,lt);s.ffn=nn.Sequential(BL(DM,DM*4),nn.GELU(),BL(DM*4,DM))
    def forward(s,x):x=x+s.attn(s.ln1(x));x=x+s.ffn(s.ln2(x));return x

class M(nn.Module):
    def __init__(s):
        super().__init__();s.emb=nn.Embedding(V,DM);s.blocks=nn.ModuleList([BLK(i)for i in range(NL)])
        s.ln=nn.LayerNorm(DM);s.head=BL(DM,V)
    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))

ck=torch.load('checkpoints/mbat_v4_final.pt',map_location='cuda',weights_only=False)
sd=ck['model']if'model'in ck else ck
m=M().to(device);m.load_state_dict(sd,strict=False);m.eval();print(f'Model loaded ({len(sd)} keys)')

tok={};data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r')
with open('checkpoints/rwkv_vocab_v20230424.txt')as f:
    for l in f:
        p=l.strip().split(' ')
        if len(p)>=2:
            try:tok[int(p[0])]=ast.literal_eval(p[1])
            except:tok[int(p[0])]=p[1]
for r in range(3):
    p_=np.random.randint(0,len(data)-80)
    x=torch.from_numpy(data[p_:p_+32].copy()).long().unsqueeze(0).to(device)
    for _ in range(40):
        p=torch.softmax(m(x)[:,-1].float()/0.8,-1);p[0,0]=0
        x=torch.cat([x,torch.multinomial(p,1)],1)
    t=''.join(tok.get(int(i),'?')for i in x[0].tolist())
    print(f'--- Run {r+1} ---');print(t[:200])
