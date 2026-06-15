"""12L纯WKV + AR CE — 云训练。编译archive的CUDA内核加速。"""
import os,sys,time,shutil
import torch,torch.nn as nn,torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Ensure CUDA_HOME for kernel compilation
if 'CUDA_HOME' not in os.environ and shutil.which('nvcc'):
    os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(shutil.which('nvcc')))

DEVICE='cuda';DM=384;VOCAB=65536;N_LAYERS=12;N=64;H=DM//N
SEQ,BSZ,LR=128,4,3e-4;N_STEPS=15000
CKPT_DIR='checkpoints';os.makedirs(CKPT_DIR,exist_ok=True)

# ── Compile CUDA kernel from local kernels/ ──
if not os.path.exists('kernels') or not os.path.exists('kernels/wkv7_fp32.cu'):
    raise FileNotFoundError('Need kernels/ directory with wkv7_fp32.cu/.cpp')

print('Compiling CUDA kernel...')
from torch.utils.cpp_extension import load
# Kernel's TORCH_LIBRARY name is "wind_backstepping"
wkv7=load(name='wind_backstepping',
    sources=['kernels/wkv7_fp32.cu','kernels/wkv7_fp32.cpp'],
    extra_cuda_cflags=['-res-usage',f'-D_C_={N}',f'-D_CHUNK_LEN_=16','-O3'])
# Compiled as TorchScript library, call via torch.ops.wind_backstepping.forward
print('Kernel compiled.')

class CUDA_WKV(nn.Module):
    def __init__(self):
        super().__init__();C=DM
        self.tm_k=nn.Parameter(torch.ones(C));self.tm_v=nn.Parameter(torch.ones(C))
        self.tm_r=nn.Parameter(torch.ones(C))
        self.key=nn.Linear(C,C,bias=False);self.val=nn.Linear(C,C,bias=False)
        self.rec=nn.Linear(C,C,bias=False);self.out=nn.Linear(C,C,bias=False)
        # Per-dim decay with head grouping
        self.w0=nn.Parameter(torch.zeros(H,N))
        for m in[self.key,self.val,self.rec,self.out]:nn.init.xavier_uniform_(m.weight,0.5)
    
    def forward(self,x):
        B,T,C=x.shape
        mk,mv,mr=torch.sigmoid(self.tm_k),torch.sigmoid(self.tm_v),torch.sigmoid(self.tm_r)
        xk=x*mk+F.pad(x[:,1:],(0,0,0,1))*(1-mk)
        xv=x*mv+F.pad(x[:,1:],(0,0,0,1))*(1-mv)
        xr=x*mr+F.pad(x[:,1:],(0,0,0,1))*(1-mr)
        k=self.key(xk);v=self.val(xv);r=torch.sigmoid(self.rec(xr))
        
        # Reshape to [B,T,H,N]
        r=r.view(B,T,H,N);k=k.view(B,T,H,N);v=v.view(B,T,H,N)
        
        # Decay: w = -softplus(w0)  (in log space, kernel handles exp)
        w=F.softplus(self.w0).view(1,1,H,N).expand(B,T,H,N).contiguous()
        
        # Kernel expects q=r, z=0, a=0 for our simplified version
        q=r.contiguous();z=torch.zeros_like(q);a=torch.zeros_like(q)
        
        # Pad T to multiples of CHUNK_LEN (16)
        pad=(16-T%16)%16
        if pad:
            def pad_last(x):return F.pad(x,(0,0,0,0,0,0,0,pad))
            q,k,v,w,z,a=map(pad_last,[q,k,v,w,z,a])
        
        # Run kernel
        TT=q.size(1)
        y=torch.empty(B,TT,H,N,device=DEVICE,dtype=torch.float32)
        s=torch.zeros(B,H,(TT+15)//16,N,N,device=DEVICE,dtype=torch.float32)
        sa=torch.zeros(B,H,(TT+15)//16,N,N,device=DEVICE,dtype=torch.float32)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,a,y,s,sa)
        
        y=y[:,:T].contiguous().view(B,T,C)
        return self.out(y)

class FFN(nn.Module):
    def __init__(self):super().__init__();self.k=nn.Linear(DM,DM*4,bias=False);self.v=nn.Linear(DM*4,DM,bias=False)
    def forward(self,x):return self.v(torch.relu(self.k(x))**2)

class Block(nn.Module):
    def __init__(self):super().__init__();self.ln1=nn.LayerNorm(DM);self.ln2=nn.LayerNorm(DM);self.wkv=CUDA_WKV();self.ffn=FFN()
    def forward(self,x):x=x+self.wkv(self.ln1(x));x=x+self.ffn(self.ln2(x));return x

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(VOCAB,DM)
        self.blocks=nn.ModuleList([Block()for _ in range(N_LAYERS)]);self.ln=nn.LayerNorm(DM);self.head=nn.Linear(DM,VOCAB,bias=False)
    def forward(self,x):
        h=self.emb(x)
        for b in self.blocks:h=b(h)
        return self.head(self.ln(h))

# Data
data=np.load(os.path.join(CKPT_DIR,'mohe_fw_rwkv_1b.npy'),mmap_mode='r');N=data.shape[0];print(f'Data:{N/1e6:.0f}M')

def get_batch(bsz,seq):
    pos=np.random.randint(0,N-seq-1,(bsz,))
    return torch.stack([torch.from_numpy(data[p:p+seq].copy()).long()for p in pos]).to(DEVICE)

model=Model().to(DEVICE);t=sum(p.numel()for p in model.parameters())
e=model.emb.weight.numel();h=model.head.weight.numel()
print(f'Params:{t/1e6:.2f}M(emb:{e/1e6:.2f}M head:{h/1e6:.2f}M core:{(t-e-h)/1e6:.2f}M)')

opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,N_STEPS)
start_step=0
if os.path.exists(os.path.join(CKPT_DIR,'pure_12l.pt')):
    ck=torch.load(os.path.join(CKPT_DIR,'pure_12l.pt'),map_location=DEVICE)
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched']);start_step=ck['step']+1;print(f'Resumed step{start_step}')

model.train();pbar=tqdm(range(start_step,N_STEPS),initial=start_step,total=N_STEPS);t0=time.time()
for step in pbar:
    x=get_batch(BSZ,SEQ);logits=model(x)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,VOCAB),x[:,1:].reshape(-1))
    opt.zero_grad();ce.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();sched.step()
    if step%1000==0:
        with torch.no_grad():
            xv=get_batch(2,SEQ);lv=model(xv)
            ppl=torch.exp(F.cross_entropy(lv[:,:-1].reshape(-1,VOCAB),xv[:,1:].reshape(-1))).item()
        pbar.set_postfix(ce=f'{ce.item():.2f}',ppl=f'{ppl:.0f}')
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),'step':step},os.path.join(CKPT_DIR,'pure_12l.pt'))

torch.save({'model':model.state_dict()},os.path.join(CKPT_DIR,'pure_12l_final.pt'))
print(f'Done in {(time.time()-t0)/60:.1f}min')

# Generation test
model.eval()
with torch.no_grad():
    x=get_batch(1,32)
    for _ in range(40):
        logits=model(x);probs=torch.softmax(logits[:,-1].float()/0.8,-1);probs[0,0]=0
        x=torch.cat([x,torch.multinomial(probs,1)],1)
    vf=os.path.join(CKPT_DIR,'rwkv_vocab_v20230424.txt')
    if os.path.exists(vf):
        with open(vf,encoding='utf-8')as f:tok={i:l.split(' ')[0]for i,l in enumerate(f)if l.strip()}
        print(f'Gen: {"".join(tok.get(int(i),"?")for i in x[0].tolist())}')
    bgs=set()
    for i in range(len(x[0])-1):bgs.add((x[0,i].item(),x[0,i+1].item()))
    print(f'Bigrams:{len(bgs)}/{len(x[0])-1}')
print('Done.')
