"""Test official 0.1B generation."""
import os, sys, torch
os.environ['CUDA_HOME']='/home/aquama/miniconda3/envs/natalia'
os.environ['CPATH']='/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
os.environ['LD_LIBRARY_PATH']='/home/aquama/miniconda3/envs/natalia/lib'
DEVICE='cuda'
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE_DIR,'rina'))
from rwkv_v7_demo import RWKV, args, RWKV_TOKENIZER

BK=os.path.join(BASE_DIR,'rwkv7-g1d-0.1b-20260129-ctx8192.pth')
TK=os.path.join(BASE_DIR,'checkpoints','rwkv_vocab_v20230424.txt')
sd=torch.load(BK,map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
md=RWKV(args).to(DEVICE)
md.load_state_dict(sd,strict=False)
md.eval()
tk=RWKV_TOKENIZER(TK)

def gen(prompt,steps=50,temp=0.8):
    ids=tk.encode(prompt)
    x=torch.tensor([ids],dtype=torch.long,device=DEVICE)
    for _ in range(steps):
        xp=x
        pad=(16-xp.size(1)%16)%16
        if pad:
            xp=torch.cat([xp,torch.zeros(1,pad,dtype=torch.long,device=DEVICE)],1)
        l,_=md(xp,return_h=True)
        l=l[:,:x.size(1)]
        probs=torch.softmax(l[:,-1].float()/temp,-1)
        probs[0,0]=0
        nxt=torch.multinomial(probs,1)
        x=torch.cat([x,nxt],1)
    return tk.decode(x[0].tolist())

prompts=["The capital of France is","User: What is 2+2?\n\nAssistant:"]
for p in prompts:
    print(f'PROMPT: {p}')
    print(f'GEN:    {gen(p, 50, 0.8)}')
    print()
