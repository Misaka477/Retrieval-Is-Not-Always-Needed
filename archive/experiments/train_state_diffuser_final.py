"""RWKV-v7 12L + State Diffuser (after ln_out). No MoE, no kernel changes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, types, math
from tqdm import tqdm

device = 'cuda'; V, D = 65536, 768; HEAD_SIZE = 64

# ── Official WKV7 (pure PyTorch) ──
def wkv7(r, w, k, v, a, b):
    B,T,C=r.shape; H=C//HEAD_SIZE; N=HEAD_SIZE
    r=r.view(B,T,H,N).float(); k=k.view(B,T,H,N).float(); v=v.view(B,T,H,N).float()
    a=a.view(B,T,H,N).float(); b=b.view(B,T,H,N).float()
    w=torch.exp(-torch.exp(w.view(B,T,H,N).float()))
    out=torch.zeros(B,T,H,N,device=r.device); s=torch.zeros(B,H,N,N,device=r.device)
    for t in range(T):
        w_t=w[:,t,:,None,:]; a_t=a[:,t].view(B,H,N,1); b_t=b[:,t].view(B,H,1,N)
        k_t=k[:,t].view(B,H,1,N); v_t=v[:,t].view(B,H,N,1); r_t=r[:,t].view(B,H,N,1)
        s = s*w_t + (s@a_t)@b_t + k_t@v_t
        out[:,t] = (s@r_t).squeeze(-1)
    return out.view(B,T,C).to(torch.float32)

# ── Official layers ──
class TimeMix(nn.Module):
    def __init__(self,args,lid):
        super().__init__()
        self.lid=lid
        self.n_head=args.dim_att//args.head_size_a
        C=args.n_embd; H=self.n_head
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']:
            setattr(self,k,nn.Parameter(torch.empty(1,1,C)))
        self.w0=nn.Parameter(torch.empty(1,1,C))
        self.w1=nn.Parameter(torch.empty(C,64))
        self.w2=nn.Parameter(torch.empty(64,C))
        self.a0=nn.Parameter(torch.empty(1,1,C))
        self.a1=nn.Parameter(torch.empty(C,64))
        self.a2=nn.Parameter(torch.empty(64,C))
        self.v0=nn.Parameter(torch.empty(1,1,C))
        self.v1=nn.Parameter(torch.empty(C,32))
        self.v2=nn.Parameter(torch.empty(32,C))
        self.g1=nn.Parameter(torch.empty(C,128))
        self.g2=nn.Parameter(torch.empty(128,C))
        self.k_k=nn.Parameter(torch.empty(1,1,C))
        self.k_a=nn.Parameter(torch.empty(1,1,C))
        self.r_k=nn.Parameter(torch.empty(H,64))
        self.receptance=nn.Linear(C,C,bias=False)
        self.key=nn.Linear(C,C,bias=False)
        self.value=nn.Linear(C,C,bias=False)
        self.output=nn.Linear(C,C,bias=False)
        self.ln_x=nn.GroupNorm(H,C,eps=64e-5)
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
        x=wkv7(r,w,k,v,-kk,kk*a)
        x=self.ln_x(x.view(B*T,C)).view(B,T,C)
        x=x+((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(-1,keepdim=True)*v.view(B,T,H,-1)).view(B,T,C)
        x=self.output(x*g); return x,vf

class ChanMix(nn.Module):
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
        self.ln1=nn.LayerNorm(args.n_embd)
        self.ln2=nn.LayerNorm(args.n_embd)
        self.att=TimeMix(args,lid)
        self.ffn=ChanMix(args)
        if lid==0: self.ln0=nn.LayerNorm(args.n_embd)
    def forward(self,x,vf):
        if self.lid==0: x=self.ln0(x)
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
bb=RWKV(args).cuda()
sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
bb.load_state_dict(sd,strict=False); bb.eval()
for p in bb.parameters(): p.requires_grad_(False)
print(f'Backbone: {sum(p.numel()/1e6 for p in bb.parameters()):.1f}M')

# ── Diffuser (after ln_out, before head) ──
class Diffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(D*2)
        self.net=nn.Sequential(nn.Linear(D*2,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.gate=nn.Parameter(torch.zeros(1))
    def forward(self,h,logits):
        cond=torch.softmax(logits*0.05,-1)@bb.head.weight
        return h+torch.tanh(self.gate)*self.net(self.norm(torch.cat([h,cond],-1)))

diff=Diffuser().cuda(); opt=torch.optim.AdamW(diff.parameters(),lr=3e-5)
ids=torch.from_numpy(np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r'))

# ── Train ──
diff.train()
pbar=tqdm(range(500))
for bi in pbar:
    s=torch.randint(0,len(ids)-4*128,(1,)).item()
    x=ids[s:s+4*128].reshape(4,128).cuda()
    with torch.no_grad():
        logits,h=bb(x,return_h=True)
    hn=h+torch.randn_like(h)*(0.02+0.08*torch.rand(1).item())
    hp=diff(hn.view(-1,D),logits.view(-1,V))
    loss=F.mse_loss(hp,h.view(-1,D))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(diff.parameters(),1.0); opt.step()
    if bi%100==0: pbar.set_postfix(loss=f'{loss.item():.4f}')
    torch.cuda.empty_cache()
torch.save({'diff':diff.state_dict()},'checkpoints/diff_final.pt')

# ── Eval ──
diff.eval()
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

prompt='The Eiffel tower is in the city of'
base=torch.tensor([tok.encode(prompt)]).cuda()
prompt_len=base.size(1)

for label,use_d,temp in [('AR',False,0.8),('AR+Diff',True,0.8)]:
    g=base.clone()
    with torch.no_grad():
        for _ in range(32):
            l,h=bb(g,return_h=True)
            if use_d:
                h=diff(h.view(-1,D),l.view(-1,V)).view(1,-1,D)
                l=bb.head(h)
            probs=torch.softmax(l[:,-1]/temp,-1)
            g=torch.cat([g,torch.multinomial(probs,1)],1)
    text=tok.decode(g[0].tolist()[prompt_len:])
    print(f'{label}: {repr(text)}')
print('Done.')
