"""Debug analytical backward step by step."""
import torch

device = "cuda"
torch.manual_seed(0)
dm = 4
np_ = 4

h = torch.randn(1, dm, device=device)
x = torch.randn(1, dm, device=device)
p = torch.randn(np_, dm, device=device)
wa = torch.randn(dm, dm*2, device=device)
wb = torch.randn(dm, dm*2, device=device)
wg = torch.randn(dm, dm*2, device=device)
wp = torch.randn(dm, dm, device=device)
wn = torch.randn(dm, device=device)
beta = 0.5
N = dm

# Forward
c = torch.cat([h, x], dim=-1)
a = torch.sigmoid(c @ wa.T)
b = torch.sigmoid(c @ wb.T)
xp = x @ wp.T
hssm = a * h + b * xp
sc = (hssm @ p.T) * beta
an = torch.softmax(sc, dim=-1)
at = an @ p
al = torch.sigmoid(c @ wg.T)
hn = hssm + al * (at - hssm)
iv = torch.rsqrt(hn.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
mn = hn.mean(-1, keepdim=True)
hnorm = (hn - mn) * iv
d_out = torch.ones_like(hn)

# Reference: PyTorch autograd on each piece
def pt_grad(fn, inputs, idx):
    """Compute d(sum(fn)) / d(inputs[idx]) using autograd."""
    inp = [i.detach().requires_grad_() for i in inputs]
    out = fn(*inp)
    out.sum().backward()
    return inp[idx].grad

# LayerNorm backward
d_hn = d_out * 1.0  # wn=1, bn=0
ref_hn = pt_grad(lambda z: z * (z - z.mean(-1, keepdim=True)) / torch.sqrt(z.var(dim=-1, unbiased=False, keepdim=True) + 1e-5), [hn], 0)
m = d_hn.sum(-1, keepdim=True) / N
s = (d_hn * hnorm).sum(-1, keepdim=True) / N
ana_hn = (d_hn - m - hnorm * s) * iv
print(f"d_hn ref:  {ref_hn[0].tolist()}")
print(f"d_hn ana:  {ana_hn[0].tolist()}")
print(f"d_hn match: {(ref_hn - ana_hn).abs().max().item() < 1e-4}")

# h_new = hssm + al * (at - hssm)  w.r.t hssm
ref_hssm = pt_grad(lambda z: z + 0.5 * (at - z), [hssm], 0)
diff = at - hn + hn  # just use at - hssm
ana_hssm = ref_hn * (1 - al)
# wait, need to compare properly
"