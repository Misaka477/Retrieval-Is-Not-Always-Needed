"""P7: 方向预测 — 预测 h 在 16 个固定方向上的升降（sign）"""
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

# ── 方向预测 ──

class SignPredictor(nn.Module):
    """从 h_t 预测 h_{t+1} 在 N 个固定方向上的升降。"""
    def __init__(s,n_dirs=16,d=512):
        super().__init__()
        # 固定随机方向（不训练）
        dirs=torch.randn(n_dirs,d)
        s.register_buffer('dirs',dirs/dirs.norm(dim=-1,keepdim=True))
        # 预测头：h_t → N 个方向的升降 logits
        s.predictor=nn.Linear(d,n_dirs)
    def forward(s,h_t,h_next):
        # 计算真实 sign 标签
        proj_t=(h_t@s.dirs.T)  # [B*T, N]
        proj_next=(h_next@s.dirs.T)
        sign_gt=torch.sign(proj_next-proj_t).detach()  # {-1,0,+1}
        # 预测 sign logits
        sign_logits=s.predictor(h_t)  # [B*T, N]
        return sign_logits,sign_gt

def sign_loss(logits,target):
    """二分类 loss: sign ∈ {-1,+1}。忽略 0（变化为 0 的位置）。"""
    mask=(target!=0)
    if mask.sum()<1:
        return torch.tensor(0.0,device=logits.device,requires_grad=True)
    logits_m=logits[mask]
    target_m=(target[mask]>0).float()  # +1 → 1, -1 → 0
    return F.binary_cross_entropy_with_logits(logits_m,target_m)

# ── VICReg ──

def vicreg(states,gamma=0.5):
    h=states.view(-1,states.size(-1))
    std=torch.sqrt(h.var(dim=0)+1e-8);vl=F.relu(gamma-std).mean()
    hc=h-h.mean(dim=0);cov=(hc.T@hc)/(h.size(0)-1)
    od=cov[~torch.eye(cov.size(0),dtype=torch.bool,device=states.device)]
    return vl+0.1*od.pow(2).mean()

# ── 指标 ──

def structure_score(sv,SEQ):
    adj=[(i*SEQ+j,i*SEQ+j+1) for i in range(sv.size(0)) for j in range(SEQ-1)]
    h=sv.reshape(-1,512);n=F.normalize(h,dim=-1)
    ac=sum((n[i]*n[j]).sum() for i,j in adj)/max(len(adj),1)
    rc=(n*n[torch.randperm(h.size(0),device=sv.device)]).sum(-1).mean().item()
    ac=ac.item()
    return ac,rc,ac/(rc+1e-8)

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r');N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config=Config();model=Backbone(config).to(device);predictor=SignPredictor().to(device)
print(f'Backbone: {sum(p.numel() for p in model.parameters())/1e6:.2f}M + SignPredictor: {sum(p.numel() for p in predictor.parameters())/1e3:.1f}K')

opt=torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if p.dim()>=2]+[p for p in predictor.parameters() if p.dim()>=2],'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if p.dim()<2]+[p for p in predictor.parameters() if p.dim()<2],'weight_decay':0.0},
],lr=1e-4,betas=(0.9,0.95))

N_STEPS=10000;SIGN_W=0.5;TR_W=0.5
CSV_PATH=os.path.join(CKPT_DIR,'p7_log.csv')
with open(CSV_PATH,'w') as f:f.write('step,tr_l,sign_l,u_l,total,sign_acc,aco,ratio\n')

model.train();predictor.train();pbar=tqdm(range(N_STEPS));t0=time.time()
for step in pbar:
    x=get_batch();states=model(x)
    h_t=states[:,:-1].reshape(-1,512);h_next=states[:,1:].reshape(-1,512)
    # Sign 预测
    sign_logits,sign_gt=predictor(h_t,h_next)
    sign_l=sign_loss(sign_logits,sign_gt)
    # Transition loss（辅助）
    tr_l=F.mse_loss(h_t,h_next)
    # VICReg
    u_l=vicreg(states)
    loss=tr_l+SIGN_W*sign_l+0.5*u_l
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
    if step%1000==0:
        model.eval();predictor.eval()
        with torch.no_grad():
            xv=get_batch();sv=model(xv)
            sl,sg=predictor(sv[:,:-1].reshape(-1,512),sv[:,1:].reshape(-1,512))
            sa=(((sl>0)==(sg>0))[sg!=0].float().mean().item()*100)
            aco,_,ratio=structure_score(sv,SEQ)
        model.train();predictor.train()
        pbar.set_postfix(tr=f'{tr_l.item():.4f}',sl=f'{sign_l.item():.4f}',sa=f'{sa:.1f}%',ratio=f'{ratio:.4f}')
        with open(CSV_PATH,'a') as f:f.write(f'{step},{tr_l.item():.6f},{sign_l.item():.6f},{u_l.item():.6f},{loss.item():.4f},{sa:.2f},{aco:.4f},{ratio:.4f}\n')
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p7_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'model':model.state_dict(),'predictor':predictor.state_dict()},os.path.join(CKPT_DIR,'p7_final.pt'))
