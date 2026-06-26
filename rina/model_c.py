"""RINA Route C — Inertia Wave: 注意力完全替换为衰减波递推，无 KV cache，纯 O(T)"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.nn import functional as F

@dataclass
class RINA_C_Config:
    block_size: int = 1024; vocab_size: int = 50304; n_layer: int = 12; n_embd: int = 512
    d_c: int = 128; head_dim: int = 64; n_head: int = 8
    dropout: float = 0.0; bias: bool = False; use_swiglu: bool = True
    ssm_steps: int = 3  # 多步 SSM: 每 token 走 K 步状态更新

class InertiaLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.d = c.n_embd; self.d_c = c.d_c; self.n_head = c.n_head; self.d_h = c.head_dim
        self.K = c.ssm_steps
        self.w_dq = nn.Linear(self.d, self.d_c, bias=c.bias)
        self.q_norm = nn.LayerNorm(self.d_c, bias=c.bias)
        self.w_mem = nn.ModuleList([nn.Linear(self.d_c, self.n_head * self.d_h, bias=c.bias) for _ in range(self.K)])
        self.w_decay = nn.ModuleList([nn.Linear(self.d_c, self.n_head, bias=c.bias) for _ in range(self.K)])
        self.w_out = nn.Linear(self.n_head * self.d_h + self.d, self.d, bias=c.bias)
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, D = x.shape
        cq = self.q_norm(self.w_dq(x))
        if self.K > 1:
            # 多步 SSM: K 步合并为复合 decay + 复合 mem，用 parallel scan
            mems = [self.w_mem[k](cq).view(B,T,self.n_head,self.d_h) for k in range(self.K)]
            decays = [torch.sigmoid(self.w_decay[k](cq)).view(B,T,self.n_head,1) for k in range(self.K)]
            d_agg = decays[0]; m_agg = mems[0]
            for k in range(1, self.K):
                d_agg = d_agg * decays[k]
                m_agg = m_agg * decays[k] + mems[k]
            a = d_agg.expand(-1,-1,-1,self.d_h)
            ca = torch.cumprod(a, dim=1)
            b_scaled = m_agg / (ca + 1e-8)
            states = ca * torch.cumsum(b_scaled, dim=1)
            sf = states.reshape(B, T, -1)
        else:
            mem = self.w_mem[0](cq).view(B, T, self.n_head, self.d_h)
            decay = torch.sigmoid(self.w_decay[0](cq))
            a = decay.unsqueeze(-1).expand(-1,-1,-1,self.d_h)
            ca = torch.cumprod(a, dim=1)
            b_scaled = mem / (ca + 1e-8)
            states = ca * torch.cumsum(b_scaled, dim=1)
            sf = states.reshape(B, T, -1)
        return self.resid_drop(self.w_out(torch.cat([x, sf], dim=-1)))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__(); h = c.n_embd * 4 * 2 // 3 // 256 * 256
        self.w1 = nn.Linear(c.n_embd, h, bias=c.bias)
        self.w2 = nn.Linear(h, c.n_embd, bias=c.bias)
        self.w3 = nn.Linear(c.n_embd, h, bias=c.bias)
        self.dp = nn.Dropout(c.dropout)
    def forward(self, x): return self.dp(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.inertia = InertiaLayer(c)
        self.ln2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = SwiGLU(c)

    def forward(self, x):
        return x + self.inertia(self.ln1(x)) + self.mlp(self.ln2(x))

class RINA_C(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd, bias=c.bias),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=c.bias)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._iw)
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight'):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))
        core = sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()
        print(f'RINA_C: {core/1e6:.2f}M core | {sum(p.numel() for p in self.parameters())/1e6:.2f}M total')

    def _iw(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size(); assert T <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        for b in self.transformer.h:
            x = b(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            l = self.lm_head(x)
            loss = F.cross_entropy(l.view(-1, l.size(-1)), targets.reshape(-1), ignore_index=-1)
            return l, loss
        return self.lm_head(x[:, [-1], :]), None
