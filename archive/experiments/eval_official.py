"""Eval diff_mse using official RWKV model (no custom backbone)."""
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.insert(0,'.'); import os,torch,types; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch.nn as nn, torch.nn.functional as F
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
D,V=768,65536
HEAD_SIZE=64

# ── Official WKV7 pure PyTorch fallback ──
def op(r,w,k,v,a,b):
    B,T,C=r.shape; H=C//HEAD_SIZE; N=HEAD_SIZE
    r4=r.view(B,T,H,N).float(); k4=k.view(B,T,H,N).float(); v4=v.view(B,T,H,N).float()
    a4=a.view(B,T,H,N).float(); b4=b.view(B,T,H,N).float()
    w4=torch.exp(-torch.exp(w.view(B,T,H,N).float()))
    out=torch.zeros(B,T,H,N,device=r.device); s=torch.zeros(B,H,N,N,device=r.device)
    for t in range(T):
        wi=w4[:,t,:,:,None]; ai=a4[:,t,:,:,None]; bi=b4[:,t,:,None,:]
        ki=k4[:,t,:,:,None]; vi=v4[:,t,:,None,:]; ri=r4[:,t,:,:,None]
        s = s*wi + (s@ai)@bi + ki@vi
        out[:,t] = (s@ri).squeeze(-1)
    return out.view(B,T,C).to(torch.float32)

# ── Official model classes ──
class Tmix(nn.Module):
    def __init__(self,args,lid):
        super().__init__();         self.lid=lid
        self.n_head=args.dim_att//args.head_size_a
        C=args.n_embd
        H=self.n_head
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']: setattr(self,k,nn.Parameter(torch.empty(1,1,C)))
        self.w0=nn.Parameter(torch.empty(1,1,C)); self.w1=nn.Parameter(torch.empty(C,64)); self.w2=nn.Parameter(torch.empty(64,C))
        self.a0=nn.Parameter(torch.empty(1,1,C)); self.a1=nn.Parameter(torch.empty(C,64)); self.a2=nn.Parameter(torch.empty(64,C))
        self.v0=nn.Parameter(torch.empty(1,1,C)); self.v1=nn.Parameter(torch.empty(C,32)); self.v2=nn.Parameter(torch.empty(32,C))
        self.g1=nn.Parameter(torch.empty(C,128)); self.g2=nn.Parameter(torch.empty(128,C))
        self.k_k=nn.Parameter(torch.empty(1,1,C)); self.k_a=nn.Parameter(torch.empty(1,1,C)); self.r_k=nn.Parameter(torch.empty(H,64))
        self.receptance=nn.Linear(C,C,bias=False); self.key=nn.Linear(C,C,bias=False)
        self.value=nn.Linear(C,C,bias=False); self.output=nn.Linear(C,C,bias=False); self.ln_x=nn.GroupNorm(H,C,eps=64e-5)
    def forward(self,x,vf):
        B,T,C=x.shape; H=self.n_head; xx=F.pad(x[:,1:],(0,0,0,1))-x
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

class Cmix_off(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.x_k=nn.Parameter(torch.empty(1,1,args.n_embd))
        self.key=nn.Linear(args.n_embd,args.dim_ffn,bias=False)
        self.value=nn.Linear(args.dim_ffn,args.n_embd,bias=False)
    def forward(self,x): xx=F.pad(x[:,1:],(0,0,0,1))-x; k=x+xx*self.x_k; k=torch.relu(self.key(k))**2; return self.value(k)

class Block(nn.Module):
    def __init__(self,args,lid):
        super().__init__()
        self.lid=lid
        self.ln0=nn.LayerNorm(args.n_embd) if lid==0 else None
        self.ln1=nn.LayerNorm(args.n_embd)
        self.ln2=nn.LayerNorm(args.n_embd)
        self.att=Tmix(args,lid)
        self.ffn=Cmix_off(args)
    def forward(self,x,vf):
        if self.lid==0:
            x=self.ln0(x)
        xx,vf=self.att(self.ln1(x),vf); x=x+xx; x=x+self.ffn(self.ln2(x)); return x,vf

class RWKV(nn.Module):
    def __init__(self,args):
        super().__init__()
        args.dim_att=args.n_embd
        args.dim_ffn=args.n_embd*4
        self.emb=nn.Embedding(args.vocab_size,args.n_embd)
        self.blocks=nn.ModuleList([Block(args,i) for i in range(args.n_layer)])
        self.ln_out=nn.LayerNorm(args.n_embd)
        self.head=nn.Linear(args.n_embd,args.vocab_size,bias=False)
    def forward(self,x,return_h=False):
        h=self.emb(x); vf=torch.empty_like(h)
        for b in self.blocks: h,vf=b(h,vf)
        h=self.ln_out(h)
        if return_h: return self.head(h),h
        return self.head(h)

# ── Load ──
args=types.SimpleNamespace(vocab_size=65536,n_embd=768,n_layer=12,head_size_a=64)
model=RWKV(args).cuda()
sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
model.load_state_dict(sd,strict=False)
model.eval(); [p.requires_grad_(False) for p in model.parameters()]

# ── Diffuser ──
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(D*2),nn.Linear(D*2,D*2),nn.GELU(),nn.Linear(D*2,D))
    def forward(self,h,c): return self.net(torch.cat([h,c],-1))
d=Denoiser().cuda(); d.load_state_dict(torch.load('checkpoints/diff_mse.pt',weights_only=False)['denoiser']); d.eval()

# ── Eval ──
prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda()
for label,use_d in [('AR',False),('AR+Diff',True)]:
    g=p.clone()
    with torch.no_grad():
        for _ in range(32):
            l,h=model(g,return_h=True)
            if use_d:
                c=(torch.softmax(l*0.05,-1)@model.head.weight).view(-1,D)
                hc=d(h.view(-1,D),c).view(1,-1,D); l=model.head(hc)
            g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1]/0.8,-1),1)],1)
    text=tok.decode(g[0].tolist()[len(tok.encode(prompt)):])
    print(f'{label}: {repr(text)}')
print('Done.')
