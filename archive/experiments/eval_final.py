import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.insert(0,'.'); import os,torch; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch.nn as nn, torch.nn.functional as F; import time
from rina.official_model import RWKV
import types

DM=768; args=types.SimpleNamespace(n_embd=DM,n_layer=12,vocab_size=65536,head_size_a=64)
sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()): 
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
m=RWKV(args).cuda(); m.load_state_dict(sd,strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)

class Diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(DM*2)
        self.net=nn.Sequential(nn.Linear(DM*2,DM*2),nn.GELU(),nn.Linear(DM*2,DM))
        self.net[-1].weight.data.zero_(); self.net[-1].bias.data.zero_()
    def forward(self,h,c): return h+self.net(self.norm(torch.cat([h,c],-1)))

diff=Diff().cuda()
if os.path.exists('checkpoints/diff_official.pt'):
    sd2=torch.load('checkpoints/diff_official.pt',weights_only=False)['diffuser']
    # Handle potential key mismatch if saved model has gate vs no gate
    try: diff.load_state_dict(sd2)
    except: print('state dict mismatch, training incomplete')
    diff.eval()

from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

def bb(x):
    pad=(16-x.size(1)%16)%16
    if pad:
        xp=torch.cat([x,torch.zeros(1,pad,dtype=torch.long,device=x.device)],1)
        l,h=m(xp,return_h=True)
        return l[:,:x.size(1)],h[:,:x.size(1)]
    return m(x,return_h=True)

prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda(); pl=p.size(1)
print(f'Prompt: {prompt}')
for label,use_d in [('AR',False),('AR+Diff',True)]:
    g=p.clone()
    with torch.no_grad():
        for _ in range(32):
            l,h=bb(g)
            if use_d:
                c=(torch.softmax(l*0.05,-1)@m.head.weight).reshape(-1,DM)
                h=diff(h.reshape(-1,DM),c).reshape(1,-1,DM)
                l=m.head(h)
            g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1].float()/0.8,-1),1)],1)
    txt=tok.decode(g[0].tolist()[pl:])
    print(f'{label}: {repr(txt[:80])}')
print('Done.')
