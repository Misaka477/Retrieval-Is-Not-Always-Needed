import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)

args = types.SimpleNamespace()

MODEL_PATH = "rwkv7-g1d-0.1b-20260129-ctx8192.pth"

args.n_layer = 12
args.n_embd = 768
D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128

args.vocab_size = 65536

DTYPE = torch.float32

args.head_size_a = 64
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = False

MyModule = nn.Module
MyFunction = lambda fn: fn
MyStatic = lambda fn: fn

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x
        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]
        for i in reversed(range(len(sorted))):
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)
    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]
            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens
    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))
    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))
    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')
    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

tokenizer = RWKV_TOKENIZER("checkpoints/rwkv_vocab_v20230424.txt")

def RWKV7_OP(r, w, k, v, a, b):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C).to(dtype=DTYPE)

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))
        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, 64))
        self.w2 = nn.Parameter(torch.empty(64, C))
        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, 64))
        self.a2 = nn.Parameter(torch.empty(64, C))
        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, 32))
        self.v2 = nn.Parameter(torch.empty(32, C))
        self.g1 = nn.Parameter(torch.empty(C, 128))
        self.g2 = nn.Parameter(torch.empty(128, C))
        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(args.n_embd) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    def forward(self, idx, return_h=False):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        if return_h:
            return self.head(x), x
        return self.head(x)

if __name__ == '__main__':
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    
    model_params = torch.load(MODEL_PATH, map_location="cpu")

with torch.no_grad():
    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params, strict=False)
    model.eval()
    
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad_(False)

    # Generation test
    prompt = "The Eiffel tower is in the city of"
    input = tokenizer.encode(prompt)
    p = torch.tensor([input]).cuda()
    plen = len(input)
    import time
    for temp in [0.01, 0.5, 0.8]:
        g = p.clone()
        t0 = time.time()
        for _ in range(32):
            l = model(g)
            probs = F.softmax(l[:, -1].float() / max(temp, 0.01), -1)
            if temp < 0.1:
                g = torch.cat([g, l[:, -1].float().argmax(-1, keepdim=True)], 1)
            else:
                g = torch.cat([g, torch.multinomial(probs, 1)], 1)
        text = tokenizer.decode(g[0].tolist()[plen:])
        print(f'temp={temp} ({time.time()-t0:.0f}s): {repr(text)}')

    # ── Diffuser training ──
    print('\n=== Diffuser training ===')
    D = args.n_embd
    class Diffuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D*2)
            self.net = nn.Sequential(nn.Linear(D*2, D*2), nn.GELU(), nn.Linear(D*2, D))
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data.zero_()
        def forward(self, h, cond):
            return h + self.net(self.norm(torch.cat([h, cond], -1)))
    
    diff = Diffuser().cuda()
    opt = torch.optim.AdamW(diff.parameters(), lr=1e-4)
    ids = torch.from_numpy(np.load('checkpoints/mohe_fw_rwkv_1b.npy', mmap_mode='r'))
    
    diff.train()
    from tqdm import tqdm
    for bi in tqdm(range(5000)):
        s = torch.randint(0, len(ids)-4*512, (1,)).item()
        x = ids[s:s+4*512].reshape(4, 512).cuda()
        with torch.no_grad():
            l, h = model(x, return_h=True)
        sigma = 0.02 + 0.08 * torch.rand(1).item()
        hn = h + torch.randn_like(h) * sigma
        c = (torch.softmax(l*0.05, -1) @ model.head.weight).reshape(-1, D)
        hp = diff(hn.reshape(-1, D), c).reshape(4, 512, D)
        loss = F.mse_loss(hp, h.detach())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0); opt.step()
        torch.cuda.empty_cache()
    
    torch.save({'diff': diff.state_dict()}, 'checkpoints/diff_final_official.pt')
    print(f'Training done. MSE={loss.item():.6f}')
    
    # ── Eval ──
    print('\n=== AR vs AR+Diff ===')
    diff.eval()
    prompt = "User: What is the capital of France?\n\nAssistant:"
    input = tokenizer.encode(prompt)
    p = torch.tensor([input]).cuda()
    plen = len(input)
    for label, use_d in [('AR', False), ('AR+Diff', True)]:
        g = p.clone()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(48):
                l, h = model(g, return_h=True)
                if use_d:
                    c = (torch.softmax(l*0.05, -1) @ model.head.weight).reshape(-1, D)
                    h = diff(h.reshape(-1, D), c).reshape(1, -1, D)
                    l = model.head(h)
                g = torch.cat([g, torch.multinomial(torch.softmax(l[:,-1].float()/0.8,-1),1)], 1)
        text = tokenizer.decode(g[0].tolist()[plen:])
        print(f'{label} ({time.time()-t0:.0f}s): {repr(text[:150])}')
    print('\nDone.')

