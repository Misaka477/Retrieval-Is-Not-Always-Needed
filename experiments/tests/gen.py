import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math
device='cuda';DM=384;VOCAB=65536;N_LAYERS=8;N_HEADS=6;HD=DM//N_HEADS;KV=HD//2
class A(nn.Module):
    def __init__(s,mk=64,lt='cbtka'):
        super().__init__();s.mk=mk;s.lt=lt
        s.q=nn.Linear(DM,DM);s.q_norm=nn.LayerNorm(DM);s.kv_proj=nn.Linear(DM,KV*N_HEADS);s.kv_norm=nn.LayerNorm(KV*N_HEADS);s.kv_expand=nn.Linear(KV*N_HEADS,DM)
        s.v_proj=nn.Linear(DM,KV*N_HEADS);s.v_expand=nn.Linear(KV*N_HEADS,DM);s.out=nn.Linear(DM,DM);s.register_buffer('pos_emb',torch.zeros(1,512,DM))
        p=torch.arange(512).unsqueeze(1);d=torch.arange(HD//2).unsqueeze(0);pe=torch.zeros(1,512,HD)
        pe[0,:,0::2]=torch.sin(p/10000**(2*d/HD));pe[0,:,1::2]=torch.cos(p/10000**(2*d/HD));s.pos_emb[:,:,:HD]=pe
    def forward(s,x):
        B,T,D=x.shape;q=s.q_norm(s.q(x)).view(B,T,N_HEADS,HD);kl=s.kv_norm(s.kv_proj(x));k=s.kv_expand(kl).view(B,T,N_HEADS,HD);v=s.v_proj(x).view(B,T,N_HEADS,KV)
        pe=s.pos_emb[:,:T,:HD];q=(q+pe.view(1,T,1,HD)).transpose(1,2);k=(k+pe.view(1,T,1,HD)).transpose(1,2);v=v.transpose(1,2)
        sc=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(HD)
        if s.lt=='window':
            W=s.mk//2;m=torch.zeros_like(sc)
            for i in range(T):S=max(0,i-W);E=min(T,i+W+1);m[:,:,i,S:E]=1
            at=F.softmax(sc.masked_fill(m==0,float('-inf')),dim=-1)
        else:
            _,idx=torch.topk(sc,s.mk,dim=-1);m=torch.zeros_like(sc).scatter_(-1,idx,1.0)
            at=F.softmax(sc.masked_fill(m==0,float('-inf')),dim=-1)
        h=torch.matmul(at,v).transpose(1,2).contiguous().view(B,T,-1);return s.out(s.v_expand(h))
class B(nn.Module):
    def __init__(s,i):
        super().__init__();s.ln1=nn.LayerNorm(DM);s.ln2=nn.LayerNorm(DM);mk=32 if i<3 else(64 if i<6 else 96);lt='window'if i<3 else'cbtka'
        s.attn=A(mk,lt);s.ffn=nn.Sequential(nn.Linear(DM,DM*4),nn.GELU(),nn.Linear(DM*4,DM))
    def forward(s,x):return x+s.attn(s.ln1(x)),x+s.ffn(s.ln2(x))
class M(nn.Module):
    def __init__(s):
        super().__init__();s.emb=nn.Embedding(VOCAB,DM);s.blocks=nn.ModuleList([B(i)for i in range(N_LAYERS)]);s.ln_out=nn.LayerNorm(DM);s.head=nn.Linear(DM,VOCAB,bias=False)
    def forward(s,x):
        h=s.emb(x)
        for b in s.blocks:h,_=b(h)
        return s.head(s.ln_out(h))

ck=torch.load('checkpoints/mbat_final.pt',map_location='cuda',weights_only=False);m=M().to(device)
m.load_state_dict(ck['model'],strict=False);m.eval()
data=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r')
for r in range(3):
    p_=np.random.randint(0,data.shape[0]-80);x=torch.from_numpy(data[p_:p_+32].copy()).long().unsqueeze(0).to(device)
    for _ in range(60):
        p=torch.softmax(m(x)[:,-1].float()/0.8,-1);p[0,0]=0;x=torch.cat([x,torch.multinomial(p,1)],1)
    tk={i:l.split(' ')[0]for i,l in enumerate(open('checkpoints/rwkv_vocab_v20230424.txt'))if l.strip()}
    print(f'\n--- Run {r+1} ---')
    print(''.join(tk.get(int(i),'?')for i in x[0].tolist()))
    bg=set();[bg.add((x[0,i].item(),x[0,i+1].item()))for i in range(x.size(1)-1)];print(f'Bigrams:{len(bg)}/{x.size(1)-1}')
