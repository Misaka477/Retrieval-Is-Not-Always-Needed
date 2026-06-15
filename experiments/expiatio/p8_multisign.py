"""P8: 多步方向预测 — 预测 步1,2,4,8 的方向升降"""
import os,time,math,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device='cuda';CKPT_DIR='checkpoints';os.makedirs(CKPT_DIR,exist_ok=True)

@dataclass
class Config:
    block_size:int=520;vocab_size:int=65536;n_layer:int=12
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

# ── 多步方向预测 ──

STEPS=[1,2,4,8];N_DIRS=16;D=512

class MultiSignPredictor(nn.Module):
    """从 h_t 预测 步1,2,4,8 的方向升降。"""
    def __init__(s):
        super().__init__()
        dirs=torch.randn(N_DIRS,D)
        s.register_buffer('dirs',dirs/dirs.norm(dim=-1,keepdim=True))
        s.shared=nn.Linear(D,D//2)
        s.heads=nn.ModuleList([nn.Linear(D//2,N_DIRS) for _ in STEPS])
    def forward(s,h_t,all_h):
        """h_t:[B,T,D], all_h:[B,T+8,D] → 多步 sign logits + labels"""
        feat=F.relu(s.shared(h_t))
        B,T,D=h_t.shape
        labels=[]
        for si,step in enumerate(STEPS):
            h_next=all_h[:,step:step+T]
            proj_t=(h_t@(s.dirs.T))  # [B,T,N]
            proj_next=(h_next@(s.dirs.T))
            labels.append(torch.sign(proj_next-proj_t).detach())
        logits=[h(feat) for h in s.heads]
        return logits,labels

def multi_sign_loss(logits,labels):
    total=0;count=0;accs=[]
    for lg,lb in zip(logits,labels):
        mask=(lb!=0)
        if mask.sum()<1:continue
        lm=lg[mask];tm=(lb[mask]>0).float()
        total+=F.binary_cross_entropy_with_logits(lm,tm)
        count+=1
        with torch.no_grad():
            accs.append(((lm>0)==(tm>0)).float().mean().item())
    if count==0:return torch.tensor(0.0,device=lg.device,requires_grad=True),0.0
    return total/count,sum(accs)/count

def vicreg(states,gamma=0.5):
    h=states.view(-1,states.size(-1))
    std=torch.sqrt(h.var(dim=0)+1e-8);vl=F.relu(gamma-std).mean()
    hc=h-h.mean(dim=0);cov=(hc.T@hc)/(h.size(0)-1)
    od=cov[~torch.eye(cov.size(0),dtype=torch.bool,device=states.device)]
    return vl+0.1*od.pow(2).mean()

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r');N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-8-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ+8].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config=Config();model=Backbone(config).to(device);predictor=MultiSignPredictor().to(device)
print(f'Backbone: {sum(p.numel() for p in model.parameters())/1e6:.2f}M + Predictor: {sum(p.numel() for p in predictor.parameters())/1e3:.1f}K')

opt=torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if p.dim()>=2]+[p for p in predictor.parameters() if p.dim()>=2],'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if p.dim()<2]+[p for p in predictor.parameters() if p.dim()<2],'weight_decay':0.0},
],lr=1e-4,betas=(0.9,0.95))

N_STEPS=10000;SIGN_W=0.5
CSV_PATH=os.path.join(CKPT_DIR,'p8_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,sign_l,sign_acc,ratio,tr_l,u_l\n')

model.train();predictor.train();pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch();states=model(x[:,:SEQ])
    h_t=states[:,:-8]
    all_h=states
    logits,labels=predictor(h_t,states)
    sign_l,sign_acc=multi_sign_loss(logits,labels)
    tr_l=F.mse_loss(h_t,states[:,1:-7])
    u_l=vicreg(states)
    loss=SIGN_W*sign_l+tr_l+0.5*u_l
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
    if step%1000==0:
        model.eval();predictor.eval()
        with torch.no_grad():
            xv=get_batch();sv=model(xv[:,:SEQ])
            lg,lb=predictor(sv[:,:-8],sv)
            _,va=multi_sign_loss(lg,lb)
            adj=[(i*SEQ+j,i*SEQ+j+1) for i in range(BSZ) for j in range(SEQ-1)]
            h=sv.reshape(-1,512);n=F.normalize(h,dim=-1)
            ac=sum((n[i]*n[j]).sum() for i,j in adj)/max(len(adj),1)
            rc=(n*n[torch.randperm(h.size(0),device=sv.device)]).sum(-1).mean().item()
        model.train();predictor.train()
        pbar.set_postfix(sl=f'{sign_l.item():.3f}',sa=f'{sign_acc:.1f}',r=f'{(ac/(rc+1e-8)).item():.4f}')
        with open(CSV_PATH,'a') as f:f.write(f'{step},{sign_l.item():.6f},{sign_acc:.4f},{ac/(rc+1e-8):.4f},{tr_l.item():.6f},{u_l.item():.6f}\n')
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p8_{step}.pt'))

print(f'\nDone')
torch.save({'model':model.state_dict(),'predictor':predictor.state_dict()},os.path.join(CKPT_DIR,'p8_final.pt'))
