"""Clean eval: official backbone + diffuser + chat format + proper 16-align."""
import sys; sys.path.insert(0,'.')
import io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import os,torch,time; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch.nn as nn, torch.nn.functional as F, types
from rina.official_model import RWKV

# ── Load backbone ──
print('Loading backbone...')
sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
m=RWKV(types.SimpleNamespace(n_embd=768,n_layer=12,vocab_size=65536,head_size_a=64)).cuda()
m.load_state_dict(sd,strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)

def bb(x):
    """Backbone with 16-alignment padding."""
    pad=(16-x.size(1)%16)%16
    if pad:
        xp=torch.cat([x,torch.full((1,pad),0,device=x.device,dtype=torch.long)],1)
        l,h=m(xp,return_h=True); return l[:,:x.size(1)],h[:,:x.size(1)]
    return m(x,return_h=True)

# ── Diffuser ──
class Diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(768*2)
        self.net=nn.Sequential(nn.Linear(768*2,768*2),nn.GELU(),nn.Linear(768*2,768))
        self.net[-1].weight.data.zero_(); self.net[-1].bias.data.zero_()
    def forward(self,h,c): return h+self.net(self.norm(torch.cat([h,c],-1)))

diff=Diff().cuda()
if os.path.exists('checkpoints/diff_official.pt'):
    sd2=torch.load('checkpoints/diff_official.pt',weights_only=False)['diffuser']
    diff.load_state_dict(sd2,strict=False); diff.eval()
    print('Diffuser loaded (training MSE=0.002)')
else:
    print('No diffuser checkpoint — running baseline only')

# ── Tokenizer ──
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

# ── Chat format (RWKV World format) ──
prompt="User: What is the capital of France?\n\nAssistant:"
p=torch.tensor([tok.encode(prompt)]).cuda()
pl=p.size(1)

print(f'\nPrompt: {prompt}')
for label,use_d in [('AR',False),('AR+Diff',True)]:
    g=p.clone(); t0=time.time()
    with torch.no_grad():
        for i in range(48):
            l,h=bb(g)
            if use_d and diff is not None:
                c=(torch.softmax(l*0.05,-1)@m.head.weight).reshape(-1,768)
                h=diff(h.reshape(-1,768),c).reshape(1,-1,768)
                l=m.head(h)
            g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1].float()/0.8,-1),1)],1)
    txt=tok.decode(g[0].tolist()[pl:])
    print(f'{label} ({time.time()-t0:.0f}s):')
    print(f'  {repr(txt[:150])}')
    print()
