"""Verify analytical CANN-SSM backward vs PyTorch autograd."""
import torch, sys
sys.path.insert(0, "..")

device = "cuda"
torch.manual_seed(0)
dm, np_ = 4, 4

h = torch.randn(1, dm, device=device)
x = torch.randn(1, dm, device=device)
p = torch.randn(np_, dm, device=device)
wa = torch.randn(dm, dm*2, device=device); ba = torch.randn(dm, device=device)
wb = torch.randn(dm, dm*2, device=device); bb = torch.randn(dm, device=device)
wg = torch.randn(dm, dm*2, device=device); bg = torch.randn(dm, device=device)
wp = torch.randn(dm, dm, device=device); bp = torch.randn(dm, device=device)
wn = torch.randn(dm, device=device); bn = torch.randn(dm, device=device)
beta = 0.5

# PyTorch reference
h2 = h.detach().requires_grad_(True)
x2 = x.detach().requires_grad_(True)
c2 = torch.cat([h2, x2], dim=-1)
a2 = torch.sigmoid(c2 @ wa.T + ba)
b2 = torch.sigmoid(c2 @ wb.T + bb)
hs2 = a2 * h2 + b2 * (x2 @ wp.T + bp)
sc2 = (hs2 @ p.T) * beta
an2 = torch.softmax(sc2, dim=-1)
at2 = an2 @ p
al2 = torch.sigmoid(c2 @ wg.T + bg)
hn2 = hs2 + al2 * (at2 - hs2)
o2 = wn * (hn2 - hn2.mean(-1, keepdim=True)) / torch.sqrt(hn2.var(dim=-1, unbiased=False, keepdim=True) + 1e-5) + bn
o2.sum().backward()
href = h2.grad.clone()
xref = x2.grad.clone()

# Analytical backward
c = torch.cat([h, x], dim=-1)
a = torch.sigmoid(c @ wa.T + ba)
b = torch.sigmoid(c @ wb.T + bb)
xp = x @ wp.T + bp
hssm = a * h + b * xp
sc = (hssm @ p.T) * beta
an = torch.softmax(sc, dim=-1)
at = an @ p
al = torch.sigmoid(c @ wg.T + bg)
hn = hssm + al * (at - hssm)
iv = torch.rsqrt(hn.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
mn = hn.mean(-1, keepdim=True)
hnorm = (hn - mn) * iv
d_out = torch.ones_like(hn)
N = dm

d_hn = d_out * wn
m = d_hn.sum(-1, keepdim=True) / N
s = (d_hn * hnorm).sum(-1, keepdim=True) / N
d_h_new = (d_hn - m - hnorm * s) * iv

diff = at - hn
d_al = (d_h_new * diff).sum(-1, keepdim=True)
d_hssm = d_h_new * (1 - al)
d_at = d_h_new * al

d_an = an * (d_at @ p.T - (an * d_at @ p.T).sum(-1, keepdim=True))
d_sc = d_an * beta
d_hssm = d_hssm + d_sc @ p   # (1,np) @ (np,dm) = (1,dm) ✓

d_a = d_hssm * h
d_b = d_hssm * xp
d_xp = d_hssm * b
d_h_ana = d_hssm * a
d_x_ana = d_xp @ wp

d_pa = d_a * a * (1 - a)
d_h_ana = d_h_ana + d_pa @ wa[:, :dm]
d_x_ana = d_x_ana + d_pa @ wa[:, dm:]

d_pb = d_b * b * (1 - b)
d_h_ana = d_h_ana + d_pb @ wb[:, :dm]
d_x_ana = d_x_ana + d_pb @ wb[:, dm:]

d_pal = d_al * al * (1 - al)
d_h_ana = d_h_ana + d_pal @ wg[:, :dm]
d_x_ana = d_x_ana + d_pal @ wg[:, dm:]

hd = (href - d_h_ana).abs().max().item()
xd = (xref - d_x_ana).abs().max().item()
print(f"h grad: ref={href[0].tolist()}")
print(f"h grad: ana={d_h_ana[0].tolist()}")
print(f"h match: {hd < 1e-4} (diff={hd:.8f})")
print(f"x match: {xd < 1e-4} (diff={xd:.8f})")
