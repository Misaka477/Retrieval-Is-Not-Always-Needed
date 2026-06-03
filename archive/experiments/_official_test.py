# Minimal modification of official rwkv_v7_demo.py
# Changes: JIT disabled (for training compat), MODEL_PATH set, CUDA kernel disabled (slow but works), LAMBADA removed
import torch, types, os, gc, math, json, sys; sys.path.insert(0,'..')
import numpy as np; import torch.nn as nn; from torch.nn import functional as F

args = types.SimpleNamespace()
MODEL_PATH = '../rwkv7-g1d-0.1b-20260129-ctx8192.pth'
args.n_layer = 12; args.n_embd = 768
D_DECAY_LORA = 64; D_AAA_LORA = 64; D_MV_LORA = 32; D_GATE_LORA = 128
args.vocab_size = 65536; DTYPE = torch.float32
args.head_size_a = 64; HEAD_SIZE = args.head_size_a
USE_CUDA_KERNEL = False  # pure Python fallback, slow but correct

MyModule = nn.Module
MyFunction = lambda fn: fn
MyStatic = lambda fn: fn

# Tokenizer
class RWKV_TOKENIZER():
    table = []; good = []; wlen = []
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            sorted += [x]; self.idx2token[idx] = x
        self.token2idx = {v:k for k,v in self.idx2token.items()}
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]; self.wlen = [0 for i in range(256)]
        for i in reversed(range(len(sorted))):
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0]); s1 = int(s[1])
                self.table[s0][s1] += [s]; self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)
    def encodeBytes(self, src):
        src_len = len(src); tokens = []; i = 0
        while i < src_len:
            s = src[i:i+1]
            if i < src_len - 1:
                s1 = int(src[i+1]); s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i:i+self.wlen[s0]]
                    try: s = next(filter(sss.startswith, self.table[s0][s1]))
                    except: pass
            tokens.append(self.token2idx[s]); i += len(s)
        return tokens
    def decodeBytes(self, tokens): return b''.join(map(lambda i: self.idx2token[i], tokens))
    def encode(self, src): return self.encodeBytes(src.encode("utf-8"))
    def decode(self, tokens): return self.decodeBytes(tokens).decode('utf-8')

tokenizer = RWKV_TOKENIZER("../checkpoints/rwkv_vocab_v20230424.txt")

# Official WKV7 OP (pure Python fallback)
def RWKV7_OP(r, w, k, v, a, b):
    B,T,C = r.shape; H = C//HEAD_SIZE; N = HEAD_SIZE
    r = r.view(B,T,H,N).float(); k = k.view(B,T,H,N).float(); v = v.view(B,T,H,N).float()
    a = a.view(B,T,H,N).float(); b = b.view(B,T,H,N).float()
    w = torch.exp(-torch.exp(w.view(B,T,H,N).float()))
    out = torch.zeros(B,T,H,N,device=r.device); s = torch.zeros(B,H,N,N,device=r.device)
    for t in range(T):
        kk = k[:,t,:].view(B,H,1,N); rr = r[:,t,:].view(B,H,N,1)
        vv = v[:,t,:].view(B,H,N,1); aa = a[:,t,:].view(B,H,N,1); bb = b[:,t,:].view(B,H,1,N)
        s = s * w[:,t,:,None,:] + s @ aa @ bb + vv @ kk
        out[:,t,:] = (s @ rr).view(B,H,N)
    return out.view(B,T,C).to(dtype=DTYPE)

# Official layers
class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__(); self.args = args; self.layer_id = layer_id
        self.head_size = args.head_size_a; self.n_head = args.dim_att // self.head_size
        H = self.n_head; C = args.n_embd
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']: setattr(self,k,nn.Parameter(torch.empty(1,1,C)))
        self.w0 = nn.Parameter(torch.empty(1,1,C)); self.w1 = nn.Parameter(torch.empty(C,64)); self.w2 = nn.Parameter(torch.empty(64,C))
        self.a0 = nn.Parameter(torch.empty(1,1,C)); self.a1 = nn.Parameter(torch.empty(C,64)); self.a2 = nn.Parameter(torch.empty(64,C))
        self.v0 = nn.Parameter(torch.empty(1,1,C)); self.v1 = nn.Parameter(torch.empty(C,32)); self.v2 = nn.Parameter(torch.empty(32,C))
        self.g1 = nn.Parameter(torch.empty(C,128)); self.g2 = nn.Parameter(torch.empty(128,C))
        self.k_k = nn.Parameter(torch.empty(1,1,C)); self.k_a = nn.Parameter(torch.empty(1,1,C)); self.r_k = nn.Parameter(torch.empty(H,64))
        self.receptance = nn.Linear(C,C,bias=False); self.key = nn.Linear(C,C,bias=False)
        self.value = nn.Linear(C,C,bias=False); self.output = nn.Linear(C,C,bias=False); self.ln_x = nn.GroupNorm(H,C,eps=64e-5)
    def forward(self, x, v_first):
        B,T,C = x.shape; H = self.n_head; xx = F.pad(x[:,1:],(0,0,0,1)) - x
        xr = x+xx*self.x_r; xw = x+xx*self.x_w; xk = x+xx*self.x_k; xv = x+xx*self.x_v; xa = x+xx*self.x_a; xg = x+xx*self.x_g
        r = self.receptance(xr); w = -F.softplus(-(self.w0+torch.tanh(xw@self.w1)@self.w2))-0.5
        k = self.key(xk); v = self.value(xv)
        if self.layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv@self.v1)@self.v2)
        a = torch.sigmoid(self.a0 + (xa@self.a1)@self.a2); g = torch.sigmoid(xg@self.g1)@self.g2
        kk = k * self.k_k; kk = F.normalize(kk.view(B,T,H,-1),dim=-1,p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RWKV7_OP(r,w,k,v,-kk,kk*a)
        x = self.ln_x(x.view(B*T,C)).view(B,T,C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(-1,keepdim=True)*v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x*g); return x,v_first

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__(); self.args = args; self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.x_k = nn.Parameter(torch.empty(1,1,args.n_embd))
        self.key = nn.Linear(args.n_embd,args.dim_ffn,bias=False)
        self.value = nn.Linear(args.dim_ffn,args.n_embd,bias=False)
    def forward(self, x): xx = self.time_shift(x) - x; k = x+xx*self.x_k; k = torch.relu(self.key(k))**2; return self.value(k)

class Block(MyModule):
    def __init__(self,args,layer_id): super().__init__(); self.args=args; self.layer_id=layer_id
        self.ln0 = nn.LayerNorm(args.n_embd) if layer_id==0 else None
        self.ln1 = nn.LayerNorm(args.n_embd); self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args,layer_id); self.ffn = RWKV_CMix_x070(args,layer_id)
    def forward(self,x,v_first):
        if self.layer_id==0: x=self.ln0(x)
        xx,v_first=self.att(self.ln1(x),v_first); x=x+xx; x=x+self.ffn(self.ln2(x)); return x,v_first

class RWKV(nn.Module):
    def __init__(self,args): super().__init__()
        args.dim_att=args.n_embd; args.dim_ffn=args.n_embd*4
        self.emb=nn.Embedding(args.vocab_size,args.n_embd)
        self.blocks=nn.ModuleList([Block(args,i) for i in range(args.n_layer)])
        self.ln_out=nn.LayerNorm(args.n_embd); self.head=nn.Linear(args.n_embd,args.vocab_size,bias=False)
    def forward(self,idx):
        x=self.emb(idx); v_first=torch.empty_like(x)
        for b in self.blocks: x,v_first=b(x,v_first)
        return self.head(self.ln_out(x))

# ── Load and generate ──
model_params = torch.load(MODEL_PATH, map_location="cpu")
for k in list(model_params.keys()):
    if isinstance(model_params[k],torch.Tensor) and model_params[k].dtype!=torch.float32:
        model_params[k]=model_params[k].float()

model = RWKV(args).to(dtype=DTYPE).cuda()
model.load_state_dict(model_params, strict=False)
model.eval()

# Generation test
prompt = "The Eiffel tower is in the city of"
input = tokenizer.encode(prompt)
p = torch.tensor([input]).cuda(); plen = len(input)

for temp in [0.01, 0.5, 0.8, 1.0]:
    g = p.clone()
    import time; t0 = time.time()
    with torch.no_grad():
        for _ in range(32):
            l = model(g)
            probs = torch.softmax(l[:,-1].float()/max(temp,0.01), -1)
            if temp < 0.1:
                g = torch.cat([g, l[:,-1].float().argmax(-1,keepdim=True)], 1)
            else:
                g = torch.cat([g, torch.multinomial(probs, 1)], 1)
    text = tokenizer.decode(g[0].tolist()[plen:])
    print(f'temp={temp} ({time.time()-t0:.0f}s): {repr(text)}')
print('Done.')
