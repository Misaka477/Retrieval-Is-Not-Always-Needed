"""Expiatio P4: Transition loss 直接训 backbone（不冻结，不加 L1）。"""
import os, time, math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from dataclasses import dataclass

device = 'cuda'; CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)

@dataclass
class Config:
    block_size: int = 512; vocab_size: int = 65536; n_layer: int = 12
    n_head: int = 8; n_embd: int = 512; dropout: float = 0.0; bias: bool = False

# ── NanoGPT backbone（同 P0）───

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head; self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B,T,C=x.size(); q,k,v=self.c_attn(x).split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1); y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C); return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__(); h=config.n_embd*4*2//3//256*256
        self.w1=nn.Linear(config.n_embd,h,bias=config.bias)
        self.w2=nn.Linear(h,config.n_embd,bias=config.bias)
        self.w3=nn.Linear(config.n_embd,h,bias=config.bias)
    def forward(self,x): return self.w2(F.silu(self.w1(x))*self.w3(x))

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd,bias=config.bias)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd,bias=config.bias)
        self.mlp=MLP(config)
    def forward(self,x): x=x+self.attn(self.ln_1(x)); x=x+self.mlp(self.ln_2(x)); return x

class Backbone(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config
        self.wte=nn.Embedding(config.vocab_size,config.n_embd)
        self.wpe=nn.Embedding(config.block_size,config.n_embd)
        self.drop=nn.Dropout(config.dropout)
        self.h=nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f=nn.LayerNorm(config.n_embd,bias=config.bias)
        self.apply(self._iw)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
    def _iw(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0.0,0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,idx):
        B,T=idx.size(); pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        x=self.drop(self.wte(idx)+self.wpe(pos))
        for b in self.h: x=b(x)
        return self.ln_f(x)

class TransitionPredictor(nn.Module):
    def __init__(self,d=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,d//2),nn.GELU(),nn.Linear(d//2,d))
    def forward(self,h): return self.net(h)

# ── 指标 ──

def structure_score(states):
    B,T,D=states.shape; h=states.view(-1,D); n=F.normalize(h,dim=-1)
    adj=[(i*T+j,i*T+j+1) for i in range(B) for j in range(T-1)]
    adj_c=sum((n[i]*n[j]).sum() for i,j in adj)/max(len(adj),1)
    idx=torch.randperm(h.size(0),device=states.device); rand_c=(n*n[idx]).sum(-1).mean().item()
    return adj_c.item(), rand_c, rand_c/max(adj_c.item(),1e-8)

# ── 数据 ──

data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r'); N=len(data)
SEQ,BSZ=512,8
def get_batch():
    pos=np.random.randint(0,N-SEQ-1,(BSZ,))
    return torch.stack([torch.from_numpy(data[p:p+SEQ].copy()).long() for p in pos]).to(device)

# ── 训练 ──

config=Config(); model=Backbone(config).to(device)
predictor=TransitionPredictor().to(device)
print(f'Backbone: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print(f'Predictor: {sum(p.numel() for p in predictor.parameters())/1e3:.1f}K')

# 所有参数都训练（不冻结）
opt=torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if p.dim()>=2]+[p for p in predictor.parameters() if p.dim()>=2],'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if p.dim()<2]+[p for p in predictor.parameters() if p.dim()<2],'weight_decay':0.0},
], lr=1e-4, betas=(0.9,0.95))

N_STEPS,VICREG_W=10000,0.5
CSV_PATH=os.path.join(CKPT_DIR,'p4_trans_log.csv')
with open(CSV_PATH,'w') as f: f.write('step,t_loss,t_cos,ratio,pd,nd,u_loss\n')

model.train(); predictor.train(); pbar=tqdm(range(N_STEPS)); t0=time.time()
for step in pbar:
    x=get_batch(); states=model(x)
    h_t=states[:,:-1]; h_next=states[:,1:]
    h_pred=predictor(h_t)
    t_loss=F.mse_loss(h_pred,h_next)  # Transition loss（主 loss）
    
    # VICReg 防止坍缩
    h=states.view(-1,states.size(-1))
    std=torch.sqrt(h.var(dim=0)+1e-8); var_loss=F.relu(0.5-std).mean()
    h_c=h-h.mean(dim=0); cov=(h_c.T@h_c)/(h.size(0)-1)
    off_diag=cov[~torch.eye(cov.size(0),dtype=torch.bool,device=states.device)]
    cov_loss=off_diag.pow(2).mean()
    u_loss=var_loss+0.1*cov_loss
    
    loss=t_loss+VICREG_W*u_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    
    if step%1000==0:
        model.eval(); predictor.eval()
        with torch.no_grad():
            xv=get_batch(); sv=model(xv)
            hp=predictor(sv[:,:-1]); hn=sv[:,1:]
            t_cos=F.cosine_similarity(hp.reshape(-1,512),hn.reshape(-1,512)).mean().item()
            adj_c,rand_c,ratio=structure_score(sv)
            pd=(F.normalize(sv[:,:-1].reshape(-1,512),dim=-1)-F.normalize(sv[:,1:].reshape(-1,512),dim=-1)).norm(dim=-1).mean().item()
            H=sv.reshape(-1,512); H_n=F.normalize(H,dim=-1)
            nd=(H_n[:BSZ]-H_n[-BSZ:]).norm(dim=-1).mean().item()  # 取第一个和最后一个序列的对应位置
        model.train(); predictor.train()
        pbar.set_postfix(tl=f'{t_loss.item():.4f}',tc=f'{t_cos:.3f}',r=f'{ratio:.4f}',pd=f'{pd:.3f}',nd=f'{nd:.3f}')
        with open(CSV_PATH,'a') as f:
            f.write(f'{step},{t_loss.item():.6f},{t_cos:.4f},{ratio:.4f},{pd:.4f},{nd:.4f},{u_loss.item():.6f}\n')
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'step':step},os.path.join(CKPT_DIR,f'p4_{step}.pt'))

print(f'\nDone in {(time.time()-t0)/60:.1f}min')
torch.save({'model':model.state_dict(),'predictor':predictor.state_dict()},os.path.join(CKPT_DIR,'p4_final.pt'))
