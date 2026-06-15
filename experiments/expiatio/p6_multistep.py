"""P6: 多步 transition — 从 h_t 同时预测 h_{t+1}, h_{t+2}, h_{t+4}, h_{t+8}"""
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

class MultiStepPredictor(nn.Module):
    """从 h_t 同时预测  h_{t+1}, h_{t+2}, h_{t+4}, h_{t+8}"""
    def __init__(s,d=512):
        super().__init__()
        s.shared=nn.Sequential(nn.Linear(d,d//2),nn.GELU())
        s.head_1=nn.Linear(d//2,d)  # step=1
        s.head_2=nn.Linear(d//2,d)  # step=2
        s.head_4=nn.Linear(d//2,d)  # step=4
        s.head_8=nn.Linear(d//2,d)  # step=8
    def forward(s,h):
        feat=s.shared(h)
        return s.head_1(feat),s.head_2(feat),s.head_4(feat),s.head_8(feat)

# ── 指标 ──

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
    pos=np.random.randint(0,N-SEQ-9,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ+8].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config=Config();model=Backbone(config).to(device);predictor=MultiStepPredictor().to(device)
print(f'Params: backbone={sum(p.numel() for p in model.parameters())/1e6:.2f}M + predictor={sum(p.numel() for p in predictor.parameters())/1e3:.1f}K')

opt=torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if p.dim()>=2]+[p for p in predictor.parameters() if p.dim()>=2],'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if p.dim()<2]+[p for p in predictor.parameters() if p.dim()<2],'weight_decay':0.0},
],lr=1e-4,betas=(0.9,0.95))

steps=[1,2,4,8];weights=[1.0,0.5,0.25,0.125]  # 短步权重高
N_STEPS=10000
CSV_PATH=os.path.join(CKPT_DIR,'p6_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,mse_1,mse_2,mse_4,mse_8,ratio,pd,nd,u_l\n')

model.train();predictor.train();pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch();states=model(x[:,:SEQ])
    h_t=states[:,:-8]
    preds=predictor(h_t)
    losses=[]
    for i,s in enumerate(steps):
        target=states[:,s:SEQ-8+s]
        pred=preds[i][:,:SEQ-8]
        losses.append(F.mse_loss(pred,target)*weights[i])
    tr_l=sum(losses)
    u_l=vicreg(states)
    loss=tr_l+0.5*u_l
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
    if step%1000==0:
        model.eval();predictor.eval()
        with torch.no_grad():
            xv=get_batch();sv=model(xv[:,:SEQ])
            preds_e=predictor(sv[:,:-8])
            ms=[F.mse_loss(preds_e[i][:,:SEQ-8],sv[:,s:SEQ-8+s]).item() for i,s in enumerate(steps)]
            adj=[(i*SEQ+j,i*SEQ+j+1) for i in range(BSZ) for j in range(SEQ-1)]
            h=sv.reshape(-1,512);n=F.normalize(h,dim=-1)
            ac=sum((n[i]*n[j]).sum() for i,j in adj)/max(len(adj),1)
            rc=(n*n[torch.randperm(h.size(0),device=sv.device)]).sum(-1).mean().item()
            pd_t=(F.normalize(sv[:,:-1].reshape(-1,512),dim=-1)-F.normalize(sv[:,1:].reshape(-1,512),dim=-1)).norm(dim=-1).mean().item()
            nd_t=(F.normalize(sv[:,:1].reshape(-1,512),dim=-1)-F.normalize(sv[:,-1:].reshape(-1,512),dim=-1)).norm(dim=-1).mean().item()
        model.train();predictor.train()
        pbar.set_postfix(m1=f'{ms[0]:.4f}',m2=f'{ms[1]:.4f}',m4=f'{ms[2]:.4f}',m8=f'{ms[3]:.4f}',r=f'{ac/(rc+1e-8):.4f}')
        with open(CSV_PATH,'a') as f:
            f.write(f'{step},{ms[0]:.6f},{ms[1]:.6f},{ms[2]:.6f},{ms[3]:.6f},{ac/(rc+1e-8):.4f},{pd_t:.4f},{nd_t:.4f},{u_l.item():.6f}\n')
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p6_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'model':model.state_dict(),'predictor':predictor.state_dict()},os.path.join(CKPT_DIR,'p6_final.pt'))
