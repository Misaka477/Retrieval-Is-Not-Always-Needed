"""P10: 翻译头在 P8 backbone 上（方向预测）。"""
import os,time,math,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device='cuda';CKPT_DIR='checkpoints';os.makedirs(CKPT_DIR,exist_ok=True)

@dataclass
class Config:
    block_size:int=520;vocab_size:int=65536;n_layer:int=12;n_head:int=8;n_embd:int=512;dropout:float=0.0;bias:bool=False

# ── 与 P8 完全一致的 backbone ──

class CausalSelfAttention(nn.Module):
    def __init__(s,c):
        super().__init__();s.c_attn=nn.Linear(c.n_embd,3*c.n_embd,bias=c.bias);s.c_proj=nn.Linear(c.n_embd,c.n_embd,bias=c.bias);s.n_head=c.n_head;s.n_embd=c.n_embd
        s.register_buffer("bias",torch.tril(torch.ones(c.block_size,c.block_size)).view(1,1,c.block_size,c.block_size))
    def forward(s,x):
        B,T,C=x.size();q,k,v=s.c_attn(x).split(C,dim=2)
        k=k.view(B,T,s.n_head,C//s.n_head).transpose(1,2);q=q.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        v=v.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        a=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        a=a.masked_fill(s.bias[:,:,:T,:T]==0,float('-inf'));a=F.softmax(a,dim=-1)
        y=(a@v).transpose(1,2).contiguous().view(B,T,C);return s.c_proj(y)

class MLP(nn.Module):
    def __init__(s,c):
        super().__init__();h=c.n_embd*4*2//3//256*256;s.w1=nn.Linear(c.n_embd,h,bias=c.bias);s.w2=nn.Linear(h,c.n_embd,bias=c.bias);s.w3=nn.Linear(c.n_embd,h,bias=c.bias)
    def forward(s,x):return s.w2(F.silu(s.w1(x))*s.w3(x))

class Block(nn.Module):
    def __init__(s,c):super().__init__();s.ln_1=nn.LayerNorm(c.n_embd,bias=c.bias);s.attn=CausalSelfAttention(c);s.ln_2=nn.LayerNorm(c.n_embd,bias=c.bias);s.mlp=MLP(c)
    def forward(s,x):x=x+s.attn(s.ln_1(x));x=x+s.mlp(s.ln_2(x));return x

class Backbone(nn.Module):
    def __init__(s,c):
        super().__init__();s.wte=nn.Embedding(c.vocab_size,c.n_embd);s.wpe=nn.Embedding(c.block_size,c.n_embd)
        s.h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]);s.ln_f=nn.LayerNorm(c.n_embd,bias=c.bias)
    def forward(s,x):
        B,T=x.size();p=torch.arange(0,T,dtype=torch.long,device=x.device);x=s.wte(x)+s.wpe(p)
        for b in s.h:x=b(x);return s.ln_f(x)

class Head(nn.Module):
    def __init__(s):super().__init__();s.net=nn.Sequential(nn.Linear(512,1024),nn.GELU(),nn.Linear(1024,65536,bias=False))
    def forward(s,h):return s.net(h)

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r');N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

c=Config()
bk=Backbone(c).to(device)
ck=torch.load(os.path.join(CKPT_DIR,'p8_final.pt'),map_location=device,weights_only=False)
bk.load_state_dict(ck['model'])
bk.eval()
for p in bk.parameters():p.requires_grad_(False)
print(f'P8 backbone: {sum(p.numel() for p in bk.parameters())/1e6:.2f}M')

head=Head().to(device)
for p in head.parameters():
    if p.dim()>=2:nn.init.normal_(p,mean=0.0,std=0.02)
print(f'Head: {sum(p.numel() for p in head.parameters())/1e6:.2f}M')

opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)
N_STEPS=5000;CSV_PATH=os.path.join(CKPT_DIR,'p10_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,ce,vce,ppl\n')

head.train();pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch()
    with torch.no_grad():h=bk(x)
    l=head(h);loss=F.cross_entropy(l[:,:-1].reshape(-1,65536),x[:,1:].reshape(-1))
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(),5.0);opt.step()
    if step%500==0:
        head.eval()
        with torch.no_grad():
            xv=get_batch();hv=bk(xv);lv=head(hv);vce=F.cross_entropy(lv[:,:-1].reshape(-1,65536),xv[:,1:].reshape(-1)).item()
        head.train()
        ppl=math.exp(vce) if vce<20 else 1e9
        pbar.set_postfix(ce=f'{loss.item():.2f}',vce=f'{vce:.2f}',ppl=f'{ppl:.0f}')
        with open(CSV_PATH,'a') as f:f.write(f'{step},{loss.item():.4f},{vce:.4f},{ppl:.0f}\n')
        torch.save({'head':head.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p10_{step}.pt'))

print(f'Done in {(time.time()-t0)/60:.1f}min')
torch.save({'head':head.state_dict()},os.path.join(CKPT_DIR,'p10_final.pt'))

# ── 生成 ──

print('\n=== Generation ===')
head.eval();bk.eval()
import ast
tok={}
with open('checkpoints/rwkv_vocab_v20230424.txt')as f:
    for l in f:
        p=l.strip().split(' ')
        if len(p)>=2:
            try:t=ast.literal_eval(p[1])
            except:t=p[1]
            if isinstance(t,bytes):t=t.decode('utf-8',errors='replace')
            tok[int(p[0])]=str(t)
for r in range(3):
    pos=np.random.randint(0,N-80);x=torch.from_numpy(data[pos:pos+32].copy()).long().unsqueeze(0).to(device)
    for _ in range(40):
        with torch.no_grad():l=head(bk(x))
        p=torch.softmax(l[:,-1].float()/0.8,-1);p[0,0]=0
        x=torch.cat([x,torch.multinomial(p,1)],1)
    t=''.join(tok.get(int(i),'?') for i in x[0].tolist())
    print(f'R{r+1}: {t[:150]}')
