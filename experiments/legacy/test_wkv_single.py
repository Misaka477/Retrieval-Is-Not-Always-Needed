import os, sys, types, torch, torch.nn as nn, torch.nn.functional as F
os.environ['CUDA_HOME']='/home/aquama/miniconda3/envs/natalia'
os.environ['LD_LIBRARY_PATH']='/home/aquama/miniconda3/envs/natalia/lib'
os.environ['CPATH']='/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
sys.path.insert(0, 'rina')
device='cuda'; DM=768; VOCAB=500; SEQ,BSZ=256,2

args=types.SimpleNamespace()
args.n_embd=DM; args.n_layer=12; args.vocab_size=VOCAB
args.dim_att=DM; args.dim_ffn=DM*4; args.head_size_a=64

from rwkv_v7_demo import RWKV_Tmix_x070
print('Kernel loaded')

class SingleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,DM)
        self.wkv=RWKV_Tmix_x070(args,0)
        self.head=nn.Linear(DM,VOCAB,bias=False)
    def forward(self,x):
        h=self.emb(x)
        h,_=self.wkv(h,torch.zeros(x.size(0),DM,device=x.device))
        return self.head(h)

m=SingleLayer().to(device)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
x=torch.randint(1,VOCAB,(BSZ,SEQ),device=device)

for i in range(500):
    l=F.cross_entropy(m(x)[:,:-1].reshape(-1,VOCAB),x[:,1:].reshape(-1))
    opt.zero_grad();l.backward();opt.step()
    if l.item()!=l.item():print(f'NaN at step {i}!');break
    if i%100==0:print(f'{i} loss={l.item():.4f}')
else:print(f'OK, final loss={l.item():.4f}')
