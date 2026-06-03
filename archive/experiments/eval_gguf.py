"""Eval AR vs AR+Diff via llama-cpp (GGUF, proven working)."""
import sys; sys.path.insert(0,'D:/Software_Development/Project/RINA_Project')
import io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import time, torch, os; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

# ── GGUF backbone via llama-cpp ──
from llama_cpp import Llama
print('Loading GGUF...')
llm=Llama(model_path='rwkv7-g1d-0.1b-20260129-ctx8192-q8_0.gguf',verbose=False,use_mmap=False)

prompt='The Eiffel tower is in the city of'
p_ids=tok.encode(prompt)
print(f'Prompt length: {len(p_ids)} tokens')

# AR baseline
t0=time.time()
out=llm(prompt,max_tokens=32,temperature=0.8,echo=False)
ar_text=out['choices'][0]['text']
print(f'AR ({time.time()-t0:.0f}s): {repr(ar_text)}')

# For AR+Diff we need the model's hidden states
# llama-cpp doesn't expose hidden states directly.
# We need the official PyTorch backbone WITH correct padding handling.
# Let's use that instead.

import types, torch.nn as nn, torch.nn.functional as F
from rina.official_model import RWKV

sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()
m=RWKV(types.SimpleNamespace(n_embd=768,n_layer=12,vocab_size=65536,head_size_a=64)).cuda()
m.load_state_dict(sd,strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)

def bb(x):
    pad=(16-x.size(1)%16)%16
    if pad:
        xp=torch.cat([x,torch.zeros(1,pad,dtype=torch.long,device=x.device)],1)
        l,h=m(xp,return_h=True); return l[:,:x.size(1)],h[:,:x.size(1)]
    return m(x,return_h=True)

class Diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(768*2)
        self.net=nn.Sequential(nn.Linear(768*2,768*2),nn.GELU(),nn.Linear(768*2,768))
        self.net[-1].weight.data.zero_(); self.net[-1].bias.data.zero_()
    def forward(self,h,c): return h+self.net(self.norm(torch.cat([h,c],-1)))

diff=Diff().cuda()
diff.load_state_dict(torch.load('checkpoints/diff_official.pt',weights_only=False)['diffuser']); diff.eval()

p=torch.tensor([p_ids]).cuda()
g=p.clone(); t0=time.time()
with torch.no_grad():
    for _ in range(32):
        l,h=bb(g)
        c=(torch.softmax(l*0.05,-1)@m.head.weight).reshape(-1,768)
        h=diff(h.reshape(-1,768),c).reshape(1,-1,768)
        l=m.head(h)
        g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1].float()/0.8,-1),1)],1)
diff_text=tok.decode(g[0].tolist()[len(p_ids):])
print(f'AR+Diff ({time.time()-t0:.0f}s): {repr(diff_text)}')
print('Done.')
