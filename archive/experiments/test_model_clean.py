import sys,os; sys.path.insert(0,'')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['RWKV_CUDA_ON']='0'
import torch,types,math,time
import torch.nn as nn, torch.nn.functional as F

HEAD_SIZE=64
DTYPE=torch.float32
args=types.SimpleNamespace(n_embd=768,n_layer=12,vocab_size=65536,head_size_a=64)

def op(r,w,k,v,a,b):
    B,T,C=r.shape; H=C//HEAD_SIZE; N=HEAD_SIZE
    r4=r.view(B,T,H,N).float(); k4=k.view(B,T,H,N).float(); v4=v.view(B,T,H,N).float()
    a4=a.view(B,T,H,N).float(); b4=b.view(B,T,H,N).float()
    w4=torch.exp(-torch.exp(w.view(B,T,H,N).float()))
    out=torch.zeros(B,T,H,N,device=r.device); s=torch.zeros(B,H,N,N,device=r.device)
    for t in range(T):
        s=s*w4[:,t,:,None,:]+(s@a4[:,t].view(B,H,N,1))@b4[:,t].view(B,H,1,N)+k4[:,t].view(B,H,1,N)@v4[:,t].view(B,H,N,1)
        out[:,t]=(s@r4[:,t].view(B,H,N,1)).view(B,H,N)
    return out.view(B,T,C).to(DTYPE)

class Tmix(nn.Module):
    def __init__(self,lid):
        super().__init__(); self.lid=lid; self.n_head=args.dim_att//args.head_size_a
        C=args.n_embd; H=self.n_head
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']: setattr(self,k,nn.Parameter(torch.empty(1,1,C)))
        self.w0=nn.Parameter(torch.empty(1,1,C)); self.w1=nn.Parameter(torch.empty(C,64)); self.w2=nn.Parameter(torch.empty(64,C))
        self.a0=nn.Parameter(torch.empty(1,1,C)); self.a1=nn.Parameter(torch.empty(C,64)); self.a2=nn.Parameter(torch.empty(64,C))
        self.v0=nn.Parameter(torch.empty(1,1,C)); self.v1=nn.Parameter(torch.empty(C,32)); self.v2=nn.Parameter(torch.empty(32,C))
        self.g1=nn.Parameter(torch.empty(C,128)); self.g2=nn.Parameter(torch.empty(128,C))
        self.k_k=nn.Parameter(torch.empty(1,1,C)); self.k_a=nn.Parameter(torch.empty(1,1,C)); self.r_k=nn.Parameter(torch.empty(H,64))
        self.receptance=nn.Linear(C,C,bias=False); self.key=nn.Linear(C,C,bias=False)
        self.value=nn.Linear(C,C,bias=False); self.output=nn.Linear(C,C,bias=False); self.ln_x=nn.GroupNorm(H,C,eps=64e-5)
    def forward(self,x,vf):
        B,T,C=x.shape; H=self.n_head
        xx=F.pad(x[:,1:],(0,0,0,1))-x
        xr=x+xx*self.x_r; xw=x+xx*self.x_w; xk=x+xx*self.x_k; xv=x+xx*self.x_v; xa=x+xx*self.x_a; xg=x+xx*self.x_g
        r=self.receptance(xr); w=-F.softplus(-(self.w0+torch.tanh(xw@self.w1)@self.w2))-0.5
        k=self.key(xk); v=self.value(xv)
        if self.lid==0: vf=v
        else: v=v+(vf-v)*torch.sigmoid(self.v0+(xv@self.v1)@self.v2)
        a=torch.sigmoid(self.a0+(xa@self.a1)@self.a2); g=torch.sigmoid(xg@self.g1)@self.g2
        kk=k*self.k_k; kk=F.normalize(kk.view(B,T,H,-1),dim=-1,p=2.0).view(B,T,C)
        k=k*(1+(a-1)*self.k_a)
        x=op(r,w,k,v,-kk,kk*a)
        x=self.ln_x(x.view(B*T,C)).view(B,T,C)
        x=x+((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(-1,keepdim=True)*v.view(B,T,H,-1)).view(B,T,C)
        x=self.output(x*g); return x,vf

class Cmix(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_k=nn.Parameter(torch.empty(1,1,args.n_embd))
        self.key=nn.Linear(args.n_embd,args.dim_ffn,bias=False)
        self.value=nn.Linear(args.dim_ffn,args.n_embd,bias=False)
    def forward(self,x): xx=F.pad(x[:,1:],(0,0,0,1))-x; k=x+xx*self.x_k; k=torch.relu(self.key(k))**2; return self.value(k)

class Block(nn.Module):
    def __init__(self,lid):
        super().__init__(); self.lid=lid
        self.ln0=nn.LayerNorm(args.n_embd) if lid==0 else None
        self.ln1=nn.LayerNorm(args.n_embd); self.ln2=nn.LayerNorm(args.n_embd)
        self.att=Tmix(lid); self.ffn=Cmix()
    def forward(self,x,vf):
        if self.lid==0: x=self.ln0(x)
        xx,vf=self.att(self.ln1(x),vf); x=x+xx; x=x+self.ffn(self.ln2(x)); return x,vf

class RWKV(nn.Module):
    def __init__(self):
        super().__init__()
        args.dim_att=args.n_embd; args.dim_ffn=args.n_embd*4
        self.emb=nn.Embedding(args.vocab_size,args.n_embd)
        self.blocks=nn.ModuleList([Block(i) for i in range(args.n_layer)])
        self.ln_out=nn.LayerNorm(args.n_embd); self.head=nn.Linear(args.n_embd,args.vocab_size,bias=False)
    def forward(self,x,return_h=False):
        h=self.emb(x); vf=torch.empty_like(h)
        for b in self.blocks: h,vf=b(h,vf)
        h=self.ln_out(h)
        if return_h: return self.head(h),h
        return self.head(h)

sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()

m=RWKV().cuda(); m.load_state_dict(sd,strict=False); m.eval()
[pp.requires_grad_(False) for pp in m.parameters()]

from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda()
print('Prompt:',prompt)

for temp in [0.01,0.5,0.8]:
    g=p.clone(); t0=time.time()
    with torch.no_grad():
        for _ in range(32):
            l=m(g); probs=torch.softmax(l[:,-1].float()/max(temp,0.01),-1)
            if temp<0.1: g=torch.cat([g,l[:,-1].float().argmax(-1,keepdim=True)],1)
            else: g=torch.cat([g,torch.multinomial(probs,1)],1)
    text=tok.decode(g[0].tolist()[len(tok.encode(prompt)):])
    print(f'temp={temp} ({time.time()-t0:.0f}s): {repr(text)}')
print('Done.')
