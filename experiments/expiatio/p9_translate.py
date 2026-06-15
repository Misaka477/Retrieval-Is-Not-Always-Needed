"""P9: 翻译头 — 在结构化状态空间上加语言头，只训头不看 backbone。"""
import os,time,math,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device='cuda';CKPT_DIR='checkpoints';os.makedirs(CKPT_DIR,exist_ok=True)

@dataclass
class Config:
    block_size:int=512;vocab_size:int=65536;n_layer:int=12
    n_head:int=8;n_embd:int=512;dropout:float=0.0;bias:bool=False

# ── NanoGPT backbone ──

class CausalSelfAttention(nn.Module):
    def __init__(s,config):
        super().__init__()
        s.c_attn=nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)
        s.c_proj=nn.Linear(config.n_embd,config.n_embd,bias=config.bias)
        s.n_head=config.n_head;s.n_embd=config.n_embd
        s.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(s,x):
        B,T,C=x.size();q,k,v=s.c_attn(x).split(s.n_embd,dim=2)
        k=k.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        q=q.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        v=v.view(B,T,s.n_head,C//s.n_head).transpose(1,2)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(s.bias[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1);y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C);return s.c_proj(y)

class MLP(nn.Module):
    def __init__(s,config):
        super().__init__();h=config.n_embd*4*2//3//256*256
        s.w1=nn.Linear(config.n_embd,h,bias=config.bias)
        s.w2=nn.Linear(h,config.n_embd,bias=config.bias)
        s.w3=nn.Linear(config.n_embd,h,bias=config.bias)
    def forward(s,x):return s.w2(F.silu(s.w1(x))*s.w3(x))

class Block(nn.Module):
    def __init__(s,config):
        super().__init__();s.ln_1=nn.LayerNorm(config.n_embd,bias=config.bias)
        s.attn=CausalSelfAttention(config);s.ln_2=nn.LayerNorm(config.n_embd,bias=config.bias)
        s.mlp=MLP(config)
    def forward(s,x):x=x+s.attn(s.ln_1(x));x=x+s.mlp(s.ln_2(x));return x

class Backbone(nn.Module):
    def __init__(s,config):
        super().__init__();s.config=config
        s.wte=nn.Embedding(config.vocab_size,config.n_embd)
        s.wpe=nn.Embedding(config.block_size,config.n_embd)
        s.drop=nn.Dropout(config.dropout)
        s.h=nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        s.ln_f=nn.LayerNorm(config.n_embd,bias=config.bias)
        s.apply(s._iw)
        for pn,p in s.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*config.n_layer))
    def _iw(s,m):
        if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0.0,0.02)
        elif isinstance(m,nn.Embedding):nn.init.normal_(m.weight,0.0,0.02)
    def forward(s,idx):
        B,T=idx.size();pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        x=s.drop(s.wte(idx)+s.wpe(pos))
        for b in s.h:x=b(x)
        return s.ln_f(x)

# ── 翻译头 ──

class LanguageHead(nn.Module):
    """轻量翻译头：状态 → token logits。两倍层 MLP 或单层 Linear。"""
    def __init__(s,d=512,hidden=1024,vocab=65536):
        super().__init__()
        s.net=nn.Sequential(
            nn.Linear(d,hidden),nn.GELU(),
            nn.Linear(hidden,vocab,bias=False),
        )
        # 缩小初始化：跟 backbone 一致
        for p in s.parameters():
            if p.dim() >= 2: nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(s, h):
        return s.net(h)

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r');N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

# 加载训练好的 backbone（冻结）
bk=Backbone(Config()).to(device)
ckpt=torch.load(os.path.join(CKPT_DIR,'p0_struct_9500.pt'),map_location=device,weights_only=False)
bk.load_state_dict(ckpt['model'])
bk.eval()
for p in bk.parameters():p.requires_grad_(False)
print(f'Backbone frozen: {sum(p.numel() for p in bk.parameters())/1e6:.2f}M')

# 初始化翻译头
head=LanguageHead().to(device)
head_params=sum(p.numel() for p in head.parameters())
print(f'Head: {head_params/1e3:.1f}K')

# 优化器（只看翻译头）
opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.01)

N_STEPS=5000
CSV_PATH=os.path.join(CKPT_DIR,'p9_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,ce,ppl,grad_norm\n')

head.train()
pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch()
    with torch.no_grad():
        states=bk(x)  # [B,T,D] backbone 冻结
    logits=head(states)  # [B,T,V]
    loss=F.cross_entropy(logits[:,:-1].reshape(-1,65536),x[:,1:].reshape(-1))
    opt.zero_grad();loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(head.parameters(),5.0)
    opt.step()
    if step%500==0:
        head.eval()
        with torch.no_grad():
            xv=get_batch();sv=bk(xv);lv=head(sv)
            ce_v=F.cross_entropy(lv[:,:-1].reshape(-1,65536),xv[:,1:].reshape(-1))
        head.train()
        ppl=math.exp(ce_v.item()) if ce_v.item()<20 else 1e9
        pbar.set_postfix(ce=f'{loss.item():.2f}',vce=f'{ce_v.item():.2f}',ppl=f'{ppl:.0f}',gn=f'{gn:.2f}')
        with open(CSV_PATH,'a') as f:f.write(f'{step},{loss.item():.4f},{ce_v.item():.4f},{gn:.4f}\n')
        torch.save({'head':head.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p9_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'head':head.state_dict()},os.path.join(CKPT_DIR,'p9_final.pt'))

# ── 生成测试 ──

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
    xb=get_batch();xb=xb[:,:32]
    for _ in range(40):
        with torch.no_grad():
            s=bk(xb);l=head(s)
        p=torch.softmax(l[:,-1].float()/0.8,-1);p[0,0]=0
        xb=torch.cat([xb,torch.multinomial(p,1)],1)
    t=''.join(tok.get(int(i),'?') for i in xb[0].tolist())
    print(f'R{r+1}: {t[:150]}')
