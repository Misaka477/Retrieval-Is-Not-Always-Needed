"""测试金字塔 coarse→fine — 纯独立，不依赖任何训练脚本"""
import os, math, time, torch, torch.nn as nn, torch.nn.functional as F

device='cuda';DM=384;NH=6;NG=2;HD=DM//NH;DC_G=HD*4//NG;DC_Q=HD*4;HR=HD//2

class BL(nn.Module):
    def __init__(s,i,o,b=False):
        super().__init__();s.w=nn.Parameter(torch.empty(o,i));s.b=nn.Parameter(torch.zeros(o))if b else None
        nn.init.normal_(s.w,0,0.02);s.g=max(1,o//128);s.s=nn.Parameter(torch.ones(s.g))
    def forward(s,x):
        w=s.w;g=s.g;wg=w.view(g,-1);sc=wg.abs().mean(-1,keepdim=True)*s.s.view(g,1)
        wq=torch.clamp(torch.round(wg/(sc+1e-8)),-1,1)*sc;wq=wq.view_as(w)
        return F.linear(x,w+(wq-w).detach(),s.b)

class PA(nn.Module):
    def __init__(s,mk=64):
        super().__init__();s.mk=mk;s.G=NG
        s.WDKV=nn.ModuleList([BL(DM,DC_G) for _ in range(s.G)])
        for g in range(s.G):s.add_module(f'kn{g}',nn.LayerNorm(DC_G))
        s.WUK=nn.ModuleList([BL(DC_G,DM//s.G) for _ in range(s.G)])
        s.WUV=nn.ModuleList([BL(DC_G,DM//s.G) for _ in range(s.G)])
        s.WDQ=BL(DM,DC_Q);s.QN=nn.LayerNorm(DC_Q);s.WUQ=BL(DC_Q,DM)
        s.WQR=BL(DC_Q,HR*NH);s.WKR=BL(DM,HR);s.WO=BL(DM,DM)
        s.register_buffer('c',torch.zeros(1,1024,HR));s.register_buffer('si',torch.zeros(1,1024,HR))
        inv=1.0/(10000**(torch.arange(0,HR,2).float()/HR));p=torch.arange(1024).float()
        s.c[0,:,0::2]=torch.cos(p[:,None]*inv[None,:]);s.c[0,:,1::2]=s.c[0,:,0::2]
        s.si[0,:,0::2]=torch.sin(p[:,None]*inv[None,:]);s.si[0,:,1::2]=s.si[0,:,0::2]
    def rope(s,x,p):c=s.c[:,p,:].unsqueeze(2);i=s.si[:,p,:].unsqueeze(2);xr=torch.cat([-x[...,1::2],x[...,0::2]],-1);return x*c+xr*i
    def coarse(s,qt,kt,st,ca,T):
        B,H,_,D=qt.shape;qc=qt[:,:,::st,:];kc=kt[:,:,::st,:];Tc=qc.size(2)
        sc=torch.matmul(qc,kc.transpose(-2,-1))/math.sqrt(D);sc=sc+ca[::st,::st].unsqueeze(0).unsqueeze(0)
        sf=torch.zeros(B,H,T,T,device=qt.device);sf[:,:,::st,::st]=sc
        sf=F.interpolate(sf.view(B*H,1,T,T),size=(T,T),mode='bilinear',align_corners=False).view(B,H,T,T)
        return sf.masked_fill(ca==float('-inf'),float('-inf'))
    def forward(s,x):
        B,T,D=x.shape;kt=min(s.mk,T)
        kv=[s.WDKV[g](x) for g in range(s.G)]  # don't norm here for speed
        kc=torch.cat([s.WUK[g](kv[g]).view(B,T,NH//s.G,HD) for g in range(s.G)],2)
        vc=torch.cat([s.WUV[g](kv[g]).view(B,T,NH//s.G,HD) for g in range(s.G)],2)
        cq=s.QN(s.WDQ(x));qc=s.WUQ(cq).view(B,T,NH,HD)
        qr=s.WQR(cq).view(B,T,NH,HR);kr=s.WKR(x).unsqueeze(2).expand(-1,-1,NH,-1)
        qr=s.rope(qr,torch.arange(T,device=x.device));kr=s.rope(kr,torch.arange(T,device=x.device))
        q=torch.cat([qc,qr],-1);k=torch.cat([kc,kr],-1);qt=q.transpose(1,2);kt_=k.transpose(1,2);vt=vc.transpose(1,2)
        ca=torch.triu(torch.full((T,T),float('-inf'),device=x.device),diagonal=1)
        # 金字塔 coarse→fine
        sc_l1=s.coarse(qt,kt_,4,ca,T)  # stride=4 coarse
        _,idx=torch.topk(sc_l1,kt,-1)
        # gather
        ik=idx.unsqueeze(-1).expand(-1,-1,-1,-1,HD+HR)
        iv=idx.unsqueeze(-1).expand(-1,-1,-1,-1,HD)
        k_exp=kt_.unsqueeze(3).expand(-1,-1,-1,kt,-1)
        v_exp=vt.unsqueeze(3).expand(-1,-1,-1,kt,-1)
        ktk=torch.gather(k_exp,2,ik)
        sc=(qt.unsqueeze(3)*ktk).sum(-1)/math.sqrt(HD+HR)
        at=F.softmax(sc,-1)
        vtk=torch.gather(v_exp,2,iv)
        h=(at.unsqueeze(-1)*vtk).sum(3)
        return s.WO(h.transpose(1,2).contiguous().view(B,T,-1))

print(f'SEQ=1024, K=64')
m=PA(mk=64).to(device)
x=torch.randn(2,1024,DM,device=device)
torch.cuda.synchronize();t0=time.time()
with torch.no_grad():o=m(x)
torch.cuda.synchronize();t1=time.time()
print(f'Out: {o.shape}')
print(f'Time: {(t1-t0)*1000:.0f}ms')
print(f'NaN: {torch.isnan(o).any().item()}')
print('✅'if not torch.isnan(o).any()else'❌')
