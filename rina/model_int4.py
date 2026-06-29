"""RINA Int4 — 纯自实现 int4 推理模型，零 dp32，零外部依赖"""
import math, torch, torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

@dataclass
class Int4Config:
    block_size: int = 512
    vocab_size: int = 128256
    n_layer: int = 16
    n_embd: int = 640
    n_head: int = 10
    n_kv_heads: int = 5
    head_dim: int = 64
    d_c: int = 160
    d_h_r: int = 32
    ssm_steps: int = 3


class Int4Linear(nn.Module):
    """纯自实现 int4 线性层。权重存为 [out, in//2] uint8 + [out] fp16 scale。
       前向时解包到 fp16 再做 matmul (全部 GPU 操作)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.register_buffer('packed', torch.empty(out_dim, in_dim // 2, dtype=torch.uint8))
        self.register_buffer('scale', torch.empty(out_dim, 1, dtype=torch.float16))

    def load_from(self, packed, scale):
        self.packed.data.copy_(packed)
        self.scale.data.copy_(scale)
        return self

    def forward(self, x):
        if not hasattr(self, '_cached_w'):
            p = self.packed.to(torch.int32)
            w = torch.stack([p & 0x0F, (p >> 4) & 0x0F], -1).reshape(self.out_dim, -1).float() - 8
            w = (w * self.scale).half()
            self._cached_w = w
        return F.linear(x, self._cached_w.to(x.dtype), None)


class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv)
        t = torch.arange(max_len, dtype=torch.float32)
        f = torch.outer(t, inv)
        self.register_buffer('cos', f.cos())
        self.register_buffer('sin', f.sin())

    def forward(self, x):
        B, H, T, D = x.shape; h = D // 2
        c = self.cos[:T].view(1, 1, T, h).expand(B, H, T, h)
        s = self.sin[:T].view(1, 1, T, h).expand(B, H, T, h)
        x0, x1 = x[..., ::2], x[..., 1::2]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)


class Int4Block(nn.Module):
    def __init__(self, c, is_sparse):
        super().__init__()
        self.is_sparse = is_sparse
        n, dc, nh, dh, nkv, dhr = c.n_embd, c.d_c, c.n_head, c.head_dim, c.n_kv_heads, c.d_h_r
        h = n * 4 * 2 // 3 // 256 * 256

        self.ln1_w = nn.Buffer(torch.empty(n, device='cuda', dtype=torch.float32))
        self.ln2_w = nn.Buffer(torch.empty(n, device='cuda', dtype=torch.float32))
        # MLP
        self.w1 = Int4Linear(n, h)
        self.w2 = Int4Linear(h, n)
        self.w3 = Int4Linear(n, h)
        if is_sparse:
            self.w_dqkv = Int4Linear(n, dc)
            self.qn_w = nn.Buffer(torch.empty(dc, device='cuda', dtype=torch.float32))
            # kn_w 未使用（原始模型中 k_norm 从未被 forward 调用）
            self.w_uq = Int4Linear(dc, nh * dh)
            self.w_uk = Int4Linear(dc, nkv * dh)
            self.w_k2v = Int4Linear(nkv * dh, nkv * dh)
            self.w_qr = Int4Linear(n, nh * dhr)
            self.w_kr = Int4Linear(n, nkv * dhr)
            self.c_proj = Int4Linear(nh * dh, n)
            self.rope = RoPE(dhr, c.block_size)
            self.rope_q = RoPE(dhr, c.block_size)
        else:
            self.w_dq = Int4Linear(n, dc)
            self.qn_w = nn.Buffer(torch.empty(dc, device='cuda', dtype=torch.float32))
            self.w_mem = nn.ModuleList([Int4Linear(dc, nh * dh) for _ in range(c.ssm_steps)])
            self.w_dec = nn.ModuleList([Int4Linear(dc, nh) for _ in range(c.ssm_steps)])
            self.w_out = Int4Linear(n + nh * dh, n)

    def set_ln(self, key_prefix, w1, w2):
        self.ln1_w.data = w1.to(device='cuda')
        self.ln2_w.data = w2.to(device='cuda')

    def forward(self, x):
        n, dc, nh, dh, nkv, dhr = 640, 160, 10, 64, 5, 32
        ln1 = F.layer_norm(x, [n], self.ln1_w, None, 1e-6)
        if self.is_sparse:
            B, T, _ = ln1.shape
            cq = F.layer_norm(self.w_dqkv(ln1), [dc], self.qn_w, None, 1e-6)
            qc = self.w_uq(cq).view(B, T, nh, dh).transpose(1, 2)
            kc = self.w_uk(cq).view(B, T, nkv, dh).transpose(1, 2)
            v = self.w_k2v(self.w_uk(cq)).view(B, T, nkv, dh).transpose(1, 2)
            qr = self.w_qr(ln1).view(B, T, nh, dhr).transpose(1, 2)
            kr = self.w_kr(ln1).view(B, T, nkv, dhr).transpose(1, 2)
            qr, kr = self.rope_q(qr), self.rope(kr)
            if nh // nkv > 1:
                r = nh // nkv
                kc = kc.repeat_interleave(r, 1)
                v = v.repeat_interleave(r, 1)
                kr = kr.repeat_interleave(r, 1)
            q = torch.cat([qc, qr], -1); k = torch.cat([kc, kr], -1)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
            out = self.c_proj(y)
        else:
            B, T, _ = ln1.shape
            cq = F.layer_norm(self.w_dq(ln1), [dc], self.qn_w, None, 1e-6)
            ms = [self.w_mem[k](cq).view(B, T, nh, dh) for k in range(3)]
            ds = [torch.sigmoid(self.w_dec[k](cq)).view(B, T, nh, 1) for k in range(3)]
            da, ma = ds[0], ms[0]
            for k in range(1, 3):
                da = da * ds[k]; ma = ma * ds[k] + ms[k]
            a = da.expand(-1, -1, -1, dh)
            ca = torch.exp(torch.cumsum(a.clamp(min=1e-10).log(), dim=1))
            ca = ca.clamp(min=1e-20)
            sf = (ca * torch.cumsum(ma / (ca + 1e-30), dim=1)).reshape(B, T, -1)
            out = self.w_out(torch.cat([ln1, sf], -1))
        r = x + out
        ln2 = F.layer_norm(r, [n], self.ln2_w, None, 1e-6)
        return r + self.w2(F.silu(self.w1(ln2)) * self.w3(ln2))


class Int4Jamba(nn.Module):
    """纯 int4 模型，从 jambaqw_complete.pt 构建，零 fp32"""
    def __init__(self, c):
        super().__init__()
        self.config = c
        n, nkv, nh, dh, dhr, dc = c.n_embd, c.n_kv_heads, c.n_head, c.head_dim, c.d_h_r, c.d_c

        # 层类型
        n_sparse = max(4, c.n_layer // 4); n_ssm = c.n_layer - n_sparse
        ratio = n_ssm // n_sparse
        ltypes = []; sc = 0
        for i in range(c.n_layer):
            if (i + 1) % (ratio + 1) == 0 and sc < n_sparse:
                ltypes.append(True); sc += 1
            else:
                ltypes.append(False)

        self.wte = nn.Embedding(c.vocab_size, n)
        self.blocks = nn.ModuleList([Int4Block(c, ltypes[i]) for i in range(c.n_layer)])
        self.ln_f_w = nn.Buffer(torch.empty(n, device='cuda', dtype=torch.float32))

    def load_export(self, path='models/out-rina-jamba-qw/jambaqw_complete.pt'):
        """从完整导出加载权重（GPU 直接）"""
        export = torch.load(path, map_location='cpu', weights_only=False)
        Q, O = export['q4'], export['other']
        cfg = export['config']
        del export

        # wte
        self.wte.weight.data.copy_(O['transformer.wte.weight'].to('cuda', dtype=torch.float32))
        # ln_f
        self.ln_f_w.data.copy_(O['transformer.ln_f.weight'].to('cuda'))

        def ld_int4(key):
            p, s = Q[key]
            return p.to('cuda'), s.to('cuda')

        for li, b in enumerate(self.blocks):
            pf = f'transformer.h.{li}.'
            b.ln1_w.data.copy_(O[pf + 'ln1.weight'].to('cuda'))
            b.ln2_w.data.copy_(O[pf + 'ln2.weight'].to('cuda'))
            # MLP
            b.w1.load_from(*ld_int4(pf + 'mlp.w1.weight'))
            b.w2.load_from(*ld_int4(pf + 'mlp.w2.weight'))
            b.w3.load_from(*ld_int4(pf + 'mlp.w3.weight'))
            if b.is_sparse:
                pfi = pf + 'path.'
                b.w_dqkv.load_from(*ld_int4(pfi + 'w_dqkv.weight'))
                b.qn_w.data.copy_(O[pfi + 'q_norm.weight'].to('cuda'))
                # k_norm 未使用
                b.w_uq.load_from(*ld_int4(pfi + 'w_uq.weight'))
                b.w_uk.load_from(*ld_int4(pfi + 'w_uk.weight'))
                b.w_k2v.load_from(*ld_int4(pfi + 'w_k2v.weight'))
                b.w_qr.load_from(*ld_int4(pfi + 'w_qr.weight'))
                b.w_kr.load_from(*ld_int4(pfi + 'w_kr.weight'))
                b.c_proj.load_from(*ld_int4(pfi + 'c_proj.weight'))
                # RoPE buffers
                for suf in ['rope.cos', 'rope.sin', 'rope_q.cos', 'rope_q.sin']:
                    buf = O.get(pfi + suf.replace('.', '.'))
                    if buf is not None:
                        mod = b.rope if 'rope_q' not in suf else b.rope_q
                        mod.register_buffer(suf.split('.')[-1], buf.to('cuda'))
            else:
                pfi = pf + 'path.'
                b.w_dq.load_from(*ld_int4(pfi + 'w_dq.weight'))
                b.qn_w.data.copy_(O[pfi + 'q_norm.weight'].to('cuda'))
                for k in range(cfg.ssm_steps):
                    b.w_mem[k].load_from(*ld_int4(pfi + f'w_mem.{k}.weight'))
                    b.w_dec[k].load_from(*ld_int4(pfi + f'w_decay.{k}.weight'))
                b.w_out.load_from(*ld_int4(pfi + 'w_out.weight'))

        del O, Q
        return self

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        x = F.layer_norm(x, [self.config.n_embd], self.ln_f_w, None, 1e-6)
        l = self.wte.weight @ x[:,[-1],:].transpose(-2,-1)
        return l.squeeze(-1)
