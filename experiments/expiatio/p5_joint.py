"""P5: L1(位置对比) + Transition(状态预测) + VICReg 联合训练。"""
import os,time,math,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device='cuda';CKPT_DIR='checkpoints';os.makedirs(CKPT_DIR,exist_ok=True)

@dataclass
class Config:
    block_size:int=512;vocab_size:int=65536;n_layer:int=12
    n_head:int=8;n_embd:int=512;dropout:float=0.0;bias:bool=False

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

class TinyMLP(nn.Module):
    def __init__(s,d=512):super().__init__();s.net=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,d))
    def forward(s,h):return s.net(h)

# ── Loss 组件 ──

def contrastive_loss(states,tau=0.5,gap=None):
    B,T,D=states.shape
    if gap is None:gap=T//4
    p=torch.randint(0,T-gap-1,(B,),device=states.device)
    a=states[torch.arange(B),p];pos=states[torch.arange(B),p+1];neg=states[torch.arange(B),p+gap]
    io=torch.randperm(B,device=states.device);ns=states[io,p]
    a=F.normalize(a,dim=-1);pos=F.normalize(pos,dim=-1);neg=F.normalize(neg+ns,dim=-1)
    pl=(a*pos).sum(-1)/tau;nl=a@neg.T/tau
    logits=torch.cat([pl.unsqueeze(-1),nl],dim=-1)
    labels=torch.zeros(B,dtype=torch.long,device=states.device)
    loss=F.cross_entropy(logits,labels)
    with torch.no_grad():
        acc=(logits.argmax(-1)==labels).float().mean().item()
        pd=(a-pos).norm(dim=-1).mean().item();nd=(a-neg).norm(dim=-1).mean().item()
    return loss,acc,pd,nd,gap

def vicreg(states,gamma=0.5):
    h=states.view(-1,states.size(-1))
    std=torch.sqrt(h.var(dim=0)+1e-8);vl=F.relu(gamma-std).mean()
    hc=h-h.mean(dim=0);cov=(hc.T@hc)/(h.size(0)-1)
    od=cov[~torch.eye(cov.size(0),dtype=torch.bool,device=states.device)]
    return vl+0.1*od.pow(2).mean()

def structure_score(states):
    B,T,D=states.shape;h=states.view(-1,D);n=F.normalize(h,dim=-1)
    adj=[(i*T+j,i*T+j+1) for i in range(B) for j in range(T-1)]
    ac=sum((n[i]*n[j]).sum() for i,j in adj)/max(len(adj),1)
    idx=torch.randperm(h.size(0),device=states.device)
    return ac.item(),(n*n[idx]).sum(-1).mean().item(),ac.item()

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r');N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config=Config();model=Backbone(config).to(device);predictor=TinyMLP().to(device)
print(f'Params: backbone={sum(p.numel() for p in model.parameters())/1e6:.2f}M + predictor={sum(p.numel() for p in predictor.parameters())/1e3:.1f}K')

opt=torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if p.dim()>=2]+[p for p in predictor.parameters() if p.dim()>=2],'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if p.dim()<2]+[p for p in predictor.parameters() if p.dim()<2],'weight_decay':0.0},
],lr=1e-4,betas=(0.9,0.95))

N_STEPS=10000
CSV_PATH=os.path.join(CKPT_DIR,'p5_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,l1_l,tr_l,u_l,total,acc,pd,nd,t_cos,ratio,lr\n')

model.train();predictor.train();pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch();states=model(x)
    # Transition
    h_pred=predictor(states[:,:-1])
    tr_l=F.mse_loss(h_pred,states[:,1:])
    # L1
    l1_l,acc,pd,nd,gap=contrastive_loss(states)
    # VICReg
    u_l=vicreg(states)
    loss=l1_l+tr_l+0.5*u_l
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
    if step%1000==0:
        model.eval();predictor.eval()
        with torch.no_grad():
            xv=get_batch();sv=model(xv);hp=predictor(sv[:,:-1])
            tc=F.cosine_similarity(hp.reshape(-1,512),sv[:,1:].reshape(-1,512)).mean().item()
            ac,rc,ratio=structure_score(sv)
        model.train();predictor.train()
        pbar.set_postfix(l1=f'{l1_l.item():.2f}',tr=f'{tr_l.item():.4f}',acc=f'{acc:.2f}',tc=f'{tc:.3f}',r=f'{ratio:.4f}')
        with open(CSV_PATH,'a') as f:f.write(f'{step},{l1_l.item():.4f},{tr_l.item():.6f},{u_l.item():.6f},{loss.item():.4f},{acc:.4f},{pd:.4f},{nd:.4f},{tc:.4f},{ratio:.4f},{opt.param_groups[0]["lr"]:.2e}\n')
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p5_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'model':model.state_dict(),'predictor':predictor.state_dict()},os.path.join(CKPT_DIR,'p5_final.pt'))
