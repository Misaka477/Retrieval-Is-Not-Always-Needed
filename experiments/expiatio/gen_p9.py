"""加载 P0 backbone + P9 head，生成文本"""
import os, ast, math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
device='cuda';D=512;V=65536;SEQ=512

# ── Backbone（同 P0）───

class CA(nn.Module):
    def __init__(s):super().__init__();s.ca=nn.Linear(D,3*D);s.cp=nn.Linear(D,D);s.nh=8;s.ed=D
    def forward(s,x):
        B,T,C=x.size();q,k,v=s.ca(x).split(C,dim=2)
        k=k.view(B,T,s.nh,C//s.nh).transpose(1,2);q=q.view(B,T,s.nh,C//s.nh).transpose(1,2);v=v.view(B,T,s.nh,C//s.nh).transpose(1,2)
        a=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        m=torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T);a=a.masked_fill(m[:T,:T]==0,float('-inf'))
        a=F.softmax(a,dim=-1);y=(a@v).transpose(1,2).contiguous().view(B,T,C);return s.cp(y)
class MLP(nn.Module):
    def __init__(s):super().__init__();h=512*4*2//3//256*256;s.w1=nn.Linear(D,h);s.w2=nn.Linear(h,D);s.w3=nn.Linear(D,h)
    def forward(s,x):return s.w2(F.silu(s.w1(x))*s.w3(x))
class Block(nn.Module):
    def __init__(s):super().__init__();s.ln1=nn.LayerNorm(D);s.attn=CA();s.ln2=nn.LayerNorm(D);s.mlp=MLP()
    def forward(s,x):x=x+s.attn(s.ln1(x));x=x+s.mlp(s.ln2(x));return x
class Backbone(nn.Module):
    def __init__(s):super().__init__();s.wte=nn.Embedding(V,D);s.wpe=nn.Embedding(SEQ,D);s.h=nn.ModuleList([Block()for _ in range(12)]);s.ln=nn.LayerNorm(D)
    def forward(s,x):
        B,T=x.size();p=torch.arange(0,T,dtype=torch.long,device=x.device);x=s.wte(x)+s.wpe(p)
        for b in s.h:x=b(x);return s.ln(x)

# ── Language Head ──

class Head(nn.Module):
    def __init__(s):super().__init__();s.net=nn.Sequential(nn.Linear(D,1024),nn.GELU(),nn.Linear(1024,V,bias=False))
    def forward(s,h):return s.net(h)

print('Loading...')
bk=Backbone().to(device)
ck=torch.load('checkpoints/p0_struct_9500.pt',map_location=device,weights_only=False)
bk.load_state_dict(ck['model'],strict=False);bk.eval()
for p in bk.parameters():p.requires_grad_(False)
head=Head().to(device)
head.load_state_dict(torch.load('checkpoints/p9_final.pt',map_location=device)['head']);head.eval()

tok={}
with open('checkpoints/rwkv_vocab_v20230424.txt')as f:
    for l in f:
        p=l.strip().split(' ')
        if len(p)>=2:
            try:t=ast.literal_eval(p[1])
            except:t=p[1]
            if isinstance(t,bytes):t=t.decode('utf-8',errors='replace')
            tok[int(p[0])]=str(t)

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r')
for r in range(3):
    pos=np.random.randint(0,len(data)-80)
    x=torch.from_numpy(data[pos:pos+32].copy()).long().unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(40):
            l=head(bk(x));p=torch.softmax(l[:,-1].float()/0.8,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    t=''.join(tok.get(int(i),'?') for i in x[0].tolist())
    print(f'R{r+1}: {t[:150]}')
