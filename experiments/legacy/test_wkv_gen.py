import os,sys,types,torch,torch.nn as nn,torch.nn.functional as F
os.environ['CUDA_HOME']='/home/aquama/miniconda3/envs/natalia'
os.environ['LD_LIBRARY_PATH']='/home/aquama/miniconda3/envs/natalia/lib'
sys.path.insert(0,'rina'); device='cuda'; DM=768; VOCAB=500; SEQ,BSZ=64,2

args=types.SimpleNamespace()
args.n_embd=DM;args.n_layer=12;args.vocab_size=VOCAB
args.dim_att=DM;args.dim_ffn=DM*4;args.head_size_a=64

from rwkv_v7_demo import RWKV_Tmix_x070,RWKV_CMix_x070

class Block(nn.Module):
    def __init__(self,i):super().__init__();self.ln1=nn.LayerNorm(DM);self.ln2=nn.LayerNorm(DM);self.wkv=RWKV_Tmix_x070(args,i);self.ffn=RWKV_CMix_x070(args,i)
    def forward(self,x,v):xx,v=self.wkv(self.ln1(x),v);x=x+xx;x=x+self.ffn(self.ln2(x));return x,v

class Model(nn.Module):
    def __init__(self):super().__init__();self.emb=nn.Embedding(VOCAB,DM);self.blocks=nn.ModuleList([Block(i)for i in range(12)]);self.head=nn.Linear(DM,VOCAB,bias=False)
    def forward(self,x):
        h=self.emb(x);v=torch.zeros(x.size(0),DM,device=x.device)
        for b in self.blocks:h,v=b(h,v)
        return self.head(h)

m=Model().to(device)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
# Train a bit
x=torch.randint(1,VOCAB,(4,64),device=device)
for i in range(2000):
    l=F.cross_entropy(m(x)[:,:-1].reshape(-1,VOCAB),x[:,1:].reshape(-1))
    opt.zero_grad();l.backward();opt.step()

# Generate autoregressively
m.eval()
with torch.no_grad():
    gen=x[:1,:32]
    for s in range(40):
        pad=(16-gen.size(1)%16)%16
        if pad:xp=torch.cat([gen,torch.zeros(1,pad,dtype=torch.long,device=device)],1)
        else:xp=gen
        logits=m(xp)
        p=F.softmax(logits[:,-1].float()/0.8,-1)
        p[0,0]=0
        if torch.isnan(p).any():print(f'NaN at step {s}!');break
        gen=torch.cat([gen,torch.multinomial(p,1)],1)
    else:print(f'OK {s+1} steps generated')
print(f'Final logits norm: {logits.norm():.2f}')
