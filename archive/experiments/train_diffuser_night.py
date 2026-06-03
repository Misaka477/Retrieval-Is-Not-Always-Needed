"""RWKV-v7 12L + State Diffuser (CUDA kernel + entropy reg).
Run: python experiments/train_diffuser_night.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, types, math
from tqdm import tqdm

device = 'cuda'; V, D = 65536, 768

# ── CUDA-backed official WKV7 (from MoHEv7) ──
from rina.model import _load_wkv7, WKV7Fn; _load_wkv7()
from rina.model_v7 import WKV7_Official

# ── Backbone: official 12L WKV7 (CUDA kernel, fast) ──
sd = torch.load('rwkv7-g1d-0.1b-20260129-ctx8192.pth', map_location='cpu', weights_only=False)
for k,v in list(sd.items()):
    if isinstance(v,torch.Tensor) and v.dtype!=torch.float32: sd[k]=v.float()

class ChanMix(nn.Module):
    def __init__(self):
        super().__init__(); C=D
        self.x_k=nn.Parameter(torch.empty(1,1,C))
        self.key=nn.Linear(C,C*4,bias=False)
        self.value=nn.Linear(C*4,C,bias=False)
    def forward(self,x):
        xx=F.pad(x[:,1:],(0,0,0,1))-x; k=x+xx*self.x_k
        k=torch.relu(self.key(k))**2; return self.value(k)

print('Loading backbone...')
wkv_layers=nn.ModuleList()
ffn_layers=nn.ModuleList()
ln1=nn.ModuleList(); ln2=nn.ModuleList()
for i in range(12):
    li=WKV7_Official(D,i)
    b=f'blocks.{i}.'
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
    wkv_layers.append(li)
    ffn=ChanMix()
    ffn.x_k.data.copy_(sd[b+'ffn.x_k']); ffn.key.weight.data.copy_(sd[b+'ffn.key.weight'])
    ffn.value.weight.data.copy_(sd[b+'ffn.value.weight']); ffn_layers.append(ffn)
    ln1.append(nn.LayerNorm(D)); ln1[-1].weight.data.copy_(sd[b+'ln1.weight']); ln1[-1].bias.data.copy_(sd[b+'ln1.bias'])
    ln2.append(nn.LayerNorm(D)); ln2[-1].weight.data.copy_(sd[b+'ln2.weight']); ln2[-1].bias.data.copy_(sd[b+'ln2.bias'])
ln0=nn.LayerNorm(D); ln0.weight.data.copy_(sd['blocks.0.ln0.weight']); ln0.bias.data.copy_(sd['blocks.0.ln0.bias'])
ln_out=nn.LayerNorm(D); ln_out.weight.data.copy_(sd['ln_out.weight']); ln_out.bias.data.copy_(sd['ln_out.bias'])
head=nn.Linear(D,V,bias=False); head.weight.data.copy_(sd['head.weight'])
for p in [*ln_out.parameters(),*head.parameters()]: p.requires_grad_(False)
wkv_layers=wkv_layers.cuda(); ffn_layers=ffn_layers.cuda()
ln1=ln1.cuda(); ln2=ln2.cuda(); ln0=ln0.cuda(); ln_out=ln_out.cuda(); head=head.cuda()
torch.cuda.empty_cache()

def backbone(x):
    h=ln0(embed(x)); vf=torch.empty_like(h)
    for i in range(12):
        h2,vf=wkv_layers[i](ln1[i](h),vf); h=h+h2
        h=h+ffn_layers[i](ln2[i](h))
    return head(ln_out(h)),ln_out(h)

embed=nn.Embedding(V,D); embed.weight.data.copy_(sd['emb.weight']); embed=embed.cuda()
for p in embed.parameters(): p.requires_grad_(False)

# ── Diffuser ──
class Diffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(D*2)
        self.net=nn.Sequential(nn.Linear(D*2,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.gate=nn.Parameter(torch.zeros(1))
    def forward(self,h,c): return h+torch.tanh(self.gate)*self.net(self.norm(torch.cat([h,c],-1)))

diff=Diffuser().cuda()
ckpt_path='checkpoints/diff_night.pt'  # resume from 50K checkpoint
if os.path.exists(ckpt_path):
    diff.load_state_dict(torch.load(ckpt_path,weights_only=False)['diff'])
    print(f'Resumed from {ckpt_path}, gate={torch.tanh(diff.gate).item():.4f}')
opt=torch.optim.AdamW(diff.parameters(),lr=3e-5)
ids=torch.from_numpy(np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r'))

N_STEPS=200000; BSZ=4; SEQ=128; LAMBDA=0.05
diff.train()
pbar=tqdm(range(N_STEPS))
for bi in pbar:
    s=torch.randint(0,len(ids)-BSZ*SEQ,(1,)).item()
    x=ids[s:s+BSZ*SEQ].reshape(BSZ,SEQ).cuda()
    with torch.no_grad():
        logits,h=backbone(x)
    
    sigma=0.02+0.08*torch.rand(1).item()
    hn=h+torch.randn_like(h)*sigma
    c=(torch.softmax(logits*0.05,-1)@head.weight).view(-1,D)
    hp=diff(hn.view(-1,D),c).view(BSZ,SEQ,D)
    loss_mse=F.mse_loss(hp,h.detach())
    
    if LAMBDA>0:
        logits_pred=head(hp)
        probs=torch.softmax(logits_pred*0.5,-1)
        entropy=-(probs*torch.log(probs.clamp(1e-10))).sum(-1).mean()
        loss=loss_mse-LAMBDA*entropy
    else:
        loss=loss_mse
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(diff.parameters(),1.0); opt.step()
    
    if bi%1000==0:
        gate_val=torch.tanh(diff.gate).item()
        pbar.set_postfix(mse=f'{loss_mse.item():.4f}',hg=gate_val)
        torch.save({'diff':diff.state_dict()},'checkpoints/diff_night_200k.pt')

torch.save({'diff':diff.state_dict()},'checkpoints/diff_night_200k.pt')
print('Done.')

# ── Quick eval ──
diff.eval()
from rina.rwkv_tokenizer import TRIE_TOKENIZER
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda()
for label,use_d,temp in [('AR',False,0.8),('AR+Diff',True,0.8)]:
    g=p.clone()
    with torch.no_grad():
        for _ in range(32):
            l,h=backbone(g)
            if use_d:
                hc=diff(h.view(-1,D),(torch.softmax(l*0.05,-1)@head.weight).view(-1,D)).view(1,-1,D)
                l=head(hc)
            g=torch.cat([g,torch.multinomial(torch.softmax(l[:,-1]/temp,-1),1)],1)
    print(f'{label}: {repr(tok.decode(g[0].tolist()[len(tok.encode(prompt)):]))}')
print('Done.')
