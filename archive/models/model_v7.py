"""MoHE-MoE v7: official WKV7 layers + MoE (2 FFN/layer, topk=1) + untied head."""
import os, torch, torch.nn as nn, torch.nn.functional as F
from .model import WKV7Fn, _load_wkv7

class WKV7_Official(nn.Module):
    """Exact RWKV_Tmix_x070. Loaded from official checkpoint, then frozen."""
    def __init__(self, dm, layer_id):
        super().__init__()
        self.layer_id = layer_id
        D, H = dm, dm // 64
        for k in ['x_r','x_w','x_k','x_v','x_a','x_g']:
            setattr(self, k, nn.Parameter(torch.empty(1,1,D)))
        self.w0 = nn.Parameter(torch.empty(1,1,D))
        self.w1 = nn.Parameter(torch.empty(D, 64))
        self.w2 = nn.Parameter(torch.empty(64, D))
        self.a0 = nn.Parameter(torch.empty(1,1,D))
        self.a1 = nn.Parameter(torch.empty(D, 64))
        self.a2 = nn.Parameter(torch.empty(64, D))
        self.v0 = nn.Parameter(torch.empty(1,1,D))
        self.v1 = nn.Parameter(torch.empty(D, 32))
        self.v2 = nn.Parameter(torch.empty(32, D))
        self.g1 = nn.Parameter(torch.empty(D, 128))
        self.g2 = nn.Parameter(torch.empty(128, D))
        self.k_k = nn.Parameter(torch.empty(1,1,D))
        self.k_a = nn.Parameter(torch.empty(1,1,D))
        self.r_k = nn.Parameter(torch.empty(H, 64))
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.receptance = nn.Linear(D,D,bias=False)
        self.key = nn.Linear(D,D,bias=False)
        self.value = nn.Linear(D,D,bias=False)
        self.output = nn.Linear(D,D,bias=False)
        self.ln_x = nn.GroupNorm(H, D, eps=64e-5)

    def forward(self, x, v_first):
        B,T,D = x.shape; H = D//64; N = 64
        xx = self.time_shift(x) - x
        xr = x + xx*self.x_r; xw = x + xx*self.x_w; xk = x + xx*self.x_k
        xv = x + xx*self.x_v; xa = x + xx*self.x_a; xg = x + xx*self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw@self.w1)@self.w2)) - 0.5
        k = self.key(xk); v = self.value(xv)
        if self.layer_id == 0: v_first = v
        else: v = v + (v_first - v)*torch.sigmoid(self.v0+(xv@self.v1)@self.v2)
        a = torch.sigmoid(self.a0+(xa@self.a1)@self.a2)
        g = torch.sigmoid(xg@self.g1)@self.g2
        kk = k*self.k_k
        kk = F.normalize(kk.view(B,T,H,-1),dim=-1,p=2.0).view(B,T,D)
        k = k*(1+(a-1)*self.k_a)

        w4d = torch.exp(-torch.exp(w.view(B,T,H,N))).contiguous()
        x = WKV7Fn.apply(
            r.view(B,T,H,N).contiguous(), w4d,
            k.view(B,T,H,N).contiguous(), v.view(B,T,H,N).contiguous(),
            -kk.view(B,T,H,N).contiguous(),
            (kk*a).view(B,T,H,N).contiguous()
        ).view(B,T,D)
        x = self.ln_x(x.view(B*T,D)).view(B,T,D)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(-1,keepdim=True)*v.view(B,T,H,-1)).view(B,T,D)
        x = self.output(x*g)
        return x, v_first

class SquaredReLU_FFN(nn.Module):
    """RWKV-7 FFN: relu(key(x+time_shift))**2 with learnable mixing."""
    def __init__(self, dm):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(1,1,dm))
        self.key = nn.Linear(dm, dm*4, bias=False)
        self.value = nn.Linear(dm*4, dm, bias=False)

    def forward(self, x):
        xx = F.pad(x[:,1:], (0,0,0,1)) - x
        k = x + xx*self.x_k
        k = torch.relu(self.key(k))**2
        return self.value(k)

    def load_official(self, sd, prefix):
        self.x_k.data.copy_(sd[prefix+'x_k'])
        self.key.weight.data.copy_(sd[prefix+'key.weight'])
        self.value.weight.data.copy_(sd[prefix+'value.weight'])

class MoHELayer(nn.Module):
    """Official WKV7 + MoE (SquaredReLU_FFN×2, topk=1)."""
    def __init__(self, dm, layer_id):
        super().__init__()
        self.wkv = WKV7_Official(dm, layer_id)
        self.ln_moe = nn.LayerNorm(dm)
        self.router = nn.Linear(dm, 2)
        # Init router bias to favor expert 0 at start (model = original RWKV)
        self.router.bias.data.fill_(-10.0)  # strongly favor expert 0
        self.experts = nn.ModuleList([SquaredReLU_FFN(dm) for _ in range(2)])

    def forward(self, h, v_first):
        h, v_first = self.wkv(h, v_first)
        h_ln = self.ln_moe(h)
        logits = self.router(h_ln)
        inds = logits.argmax(dim=-1, keepdim=True)
        exp_out = torch.where(inds==1, self.experts[1](h_ln), self.experts[0](h_ln))
        h = h + exp_out
        return h, v_first

class MoHEv7(nn.Module):
    """6× [Official WKV7 → MoE(topk=1)]. Untied head. Loads official 0.1B weights."""
    def __init__(self, vocab, dm, n_layers=6, ckpt_path=None):
        super().__init__()
        _load_wkv7()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, dm)
        self.ln0 = nn.LayerNorm(dm)
        self.layers = nn.ModuleList([MoHELayer(dm, i) for i in range(n_layers)])
        self.ln_out = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, vocab, bias=False)

        if ckpt_path and os.path.exists(ckpt_path):
            self.load_ckpt(ckpt_path)

    def load_ckpt(self, path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        self.embed.weight.data.copy_(sd['emb.weight'])
        self.ln0.weight.data.copy_(sd['blocks.0.ln0.weight'])
        self.ln0.bias.data.copy_(sd['blocks.0.ln0.bias'])
        self.head.weight.data.copy_(sd['head.weight'])
        self.ln_out.weight.data.copy_(sd['ln_out.weight'])
        self.ln_out.bias.data.copy_(sd['ln_out.bias'])

        for i, layer in enumerate(self.layers):
            off_i = i * 2 if self.n_layers <= 6 else i
            b = f'blocks.{off_i}.'
            wkv = layer.wkv
            wkv_sd = sd
            # WKV params
            for k in ['x_r','x_w','x_k','x_v','x_a','x_g']:
                getattr(wkv, k).data.copy_(wkv_sd[b+f'att.{k}'])
            for k in ['w0','w1','w2','a0','a1','a2','v0','v1','v2','g1','g2','k_k','k_a','r_k']:
                getattr(wkv, k).data.copy_(wkv_sd[b+f'att.{k}'])
            wkv.receptance.weight.data.copy_(wkv_sd[b+'att.receptance.weight'])
            wkv.key.weight.data.copy_(wkv_sd[b+'att.key.weight'])
            wkv.value.weight.data.copy_(wkv_sd[b+'att.value.weight'])
            wkv.output.weight.data.copy_(wkv_sd[b+'att.output.weight'])
            wkv.ln_x.weight.data.copy_(wkv_sd[b+'att.ln_x.weight'])
            wkv.ln_x.bias.data.copy_(wkv_sd[b+'att.ln_x.bias'])

            off_b = off_i
            layer.experts[0].load_official(sd, f'blocks.{off_b}.ffn.')
            e1_off = off_b + 1 if off_b + 1 < 12 else off_b
            layer.experts[1].load_official(sd, f'blocks.{e1_off}.ffn.')

        print(f'Loaded: WKV7 + MoE initialized from {path}')

    def forward(self, x, return_state=False):
        h = self.ln0(self.embed(x))
        v_first = torch.empty_like(h)
        for layer in self.layers:
            h, v_first = layer(h, v_first)
        h = self.ln_out(h)
        if return_state:
            return self.head(h), h
        return self.head(h)
