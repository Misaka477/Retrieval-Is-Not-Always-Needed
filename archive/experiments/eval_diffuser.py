"""Compare AR vs AR+Diff with 200K diffuser checkpoint."""
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.insert(0,'.'); import os,torch; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch.nn as nn, torch.nn.functional as F

from rina.model import _load_wkv7; _load_wkv7()
from rina.model_v7 import WKV7_Official
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

D,V=768,65536
sd=torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth',map_location='cpu',weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()

class Cmix(nn.Module):
    def __init__(self):
        super().__init__()
        C=D
        self.x_k=nn.Parameter(torch.empty(1,1,C))
        self.key=nn.Linear(C,C*4,bias=False)
        self.value=nn.Linear(C*4,C,bias=False)
    def forward(self,x):
        xx=F.pad(x[:,1:],(0,0,0,1))-x; k=x+xx*self.x_k
        k=torch.relu(self.key(k))**2; return self.value(k)

wkv,ffn,ln1,ln2=[],[],[],[]
for i in range(12):
    li=WKV7_Official(D,i).cuda(); b=f'blocks.{i}.'
    for k in ['x_r','x_w','x_k','x_v','x_a','x_g']: getattr(li,k).data.copy_(sd[b+f'att.{k}'])
    for k in ['w0','w1','w2','a0','a1','a2','v0','v1','v2','g1','g2','k_k','k_a','r_k']:
        getattr(li,k).data.copy_(sd[b+f'att.{k}'])
    li.receptance.weight.data.copy_(sd[b+'att.receptance.weight'])
    li.key.weight.data.copy_(sd[b+'att.key.weight'])
    li.value.weight.data.copy_(sd[b+'att.value.weight'])
    li.output.weight.data.copy_(sd[b+'att.output.weight'])
    li.ln_x.weight.data.copy_(sd[b+'att.ln_x.weight'])
    li.ln_x.bias.data.copy_(sd[b+'att.ln_x.bias'])
    for p in li.parameters(): p.requires_grad_(False)
    wkv.append(li)
    ci=Cmix().cuda()
    ci.x_k.data.copy_(sd[b+'ffn.x_k']); ci.key.weight.data.copy_(sd[b+'ffn.key.weight'])
    ci.value.weight.data.copy_(sd[b+'ffn.value.weight']); ffn.append(ci)
    ln1.append(nn.LayerNorm(D).cuda()); ln1[-1].weight.data.copy_(sd[b+'ln1.weight']); ln1[-1].bias.data.copy_(sd[b+'ln1.bias'])
    ln2.append(nn.LayerNorm(D).cuda()); ln2[-1].weight.data.copy_(sd[b+'ln2.weight']); ln2[-1].bias.data.copy_(sd[b+'ln2.bias'])

ln0=nn.LayerNorm(D).cuda(); ln0.weight.data.copy_(sd['blocks.0.ln0.weight']); ln0.bias.data.copy_(sd['blocks.0.ln0.bias'])
lo=nn.LayerNorm(D).cuda(); lo.weight.data.copy_(sd['ln_out.weight']); lo.bias.data.copy_(sd['ln_out.bias'])
hd=nn.Linear(D,V,bias=False).cuda(); hd.weight.data.copy_(sd['head.weight'])
emb=nn.Embedding(V,D).cuda(); emb.weight.data.copy_(sd['emb.weight'])
for p in [*lo.parameters(),*hd.parameters(),*emb.parameters()]: p.requires_grad_(False)

def bb(x):
    h=ln0(emb(x)); vf=torch.empty_like(h)
    for i in range(12):
        h2,vf=wkv[i](ln1[i](h),vf); h=h+h2; h=h+ffn[i](ln2[i](h))
    return hd(lo(h)),lo(h)

class Diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(D*2)
        self.net=nn.Sequential(nn.Linear(D*2,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.gate=nn.Parameter(torch.zeros(1))
    def forward(self,h,c): return h+torch.tanh(self.gate)*self.net(self.norm(torch.cat([h,c],-1)))

diff=Diff().cuda()
diff.load_state_dict(torch.load('checkpoints/diff_night_200k.pt',weights_only=False)['diff']); diff.eval()
print(f'gate: {torch.tanh(diff.gate).item():.4f}')

prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda()
for label,use_d in [('AR',False),('AR+Diff',True)]:
    g=p.clone()
    with torch.no_grad():
        for _ in range(64):
            l,h=bb(g)
            if use_d:
                hc=diff(h.view(-1,D),(torch.softmax(l*0.05,-1)@hd.weight).view(-1,D)).view(1,-1,D)
                l=hd(hc)
            g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1]/0.8,-1),1)],1)
    text=tok.decode(g[0].tolist()[len(tok.encode(prompt)):])
    print(f'{label}: {repr(text)}')
print('Done.')
