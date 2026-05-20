"""Piece by piece gradient comparison."""
import torch

device = "cuda"
torch.manual_seed(0)
dm = 4
np_ = 4

h = torch.randn(1, dm, device=device)
x = torch.randn(1, dm, device=device)
p = torch.randn(np_, dm, device=device)

# Compare grad_h computed analytically vs autograd for the whole cell
# Use autograd.grad for reference

def forward_all(h, x):
    wa = torch.randn(dm, dm*2, device=device)
    wb = torch.randn(dm, dm*2, device=device)
    wg = torch.randn(dm, dm*2, device=device)
    wp = torch.randn(dm, dm, device=device)
    # Fixed params - not differentiated
    return None

# Simpler approach: compute grad_via_autograd
wa = torch.randn(dm, dm*2, device=device)
wb = torch.randn(dm, dm*2, device=device)
wg = torch.randn(dm, dm*2, device=device)
wp = torch.randn(dm, dm, device=device)
wn = torch.randn(dm, device=device)
beta = 0.5

h2 = h.detach().requires_grad_()
x2 = x.detach().requires_grad_()
c2 = torch.cat([h2, x2], dim=-1)
a2 = torch.sigmoid(c2 @ wa.T)
b2 = torch.sigmoid(c2 @ wb.T)
xp2 = x2 @ wp.T
hs2 = a2 * h2 + b2 * xp2
sc2 = (hs2 @ p.T) * beta
an2 = torch.softmax(sc2, dim=-1)
at2 = an2 @ p
al2 = torch.sigmoid(c2 @ wg.T)
hn2 = hs2 + al2 * (at2 - hs2)
o2 = wn * (hn2 - hn2.mean(-1, keepdim=True)) / torch.sqrt(hn2.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
o2.sum().backward()
href = h2.grad.clone()

# Now compute analytical gradient
# Re-do forward for intermediates (no grad)
hssm = a2.detach() * h + b2.detach() * xp2.detach()
sc = (hssm @ p.T) * beta
an = torch.softmax(sc, dim=-1)
at = an @ p
al = al2.detach()
hn = hssm + al * (at - hssm)
iv = torch.rsqrt(hn.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
mn = hn.mean(-1, keepdim=True)
hnorm = (hn - mn) * iv
d_out = torch.ones_like(hn)
N = dm

# LayerNorm backward
d_hn = d_out * wn
m = d_hn.sum(-1, keepdim=True) / N
s = (d_hn * hnorm).sum(-1, keepdim=True) / N
d_h_new = (d_hn - m - hnorm * s) * iv

# h_new = hssm + al * (at - hssm)
diff = at - hn  # compute properly
diff = at - hssm
d_al = (d_h_new * diff).sum(-1, keepdim=True)
d_hssm = d_h_new * (1 - al)
d_at_contrib = d_h_new * al

# attracted = an @ p
d_an = an * (d_at_contrib @ p.T - (an * d_at_contrib @ p.T).sum(-1, keepdim=True))

# scores = (hssm @ p.T) * beta
d_sc = d_an * beta
d_hssm_attn = d_sc @ p
d_hssm = d_hssm + d_hssm_attn

# hssm = a * h + b * xp  
# a = sigmoid(c@wa.T)
# b = sigmoid(c@wb.T)
# xp = x@wp.T
d_a = d_hssm * h
d_b = d_hssm * xp2.detach()
d_xp = d_hssm * b2.detach()
d_h_ana = d_hssm * a2.detach()
d_x_ana = d_xp @ wp

# sigmoid backward for a
d_pa = d_a * a2.detach() * (1 - a2.detach())
d_h_ana += d_pa @ wa[:, :dm]
d_x_ana += d_pa @ wa[:, dm:]

# sigmoid backward for b
d_pb = d_b * b2.detach() * (1 - b2.detach())
d_h_ana += d_pb @ wb[:, :dm]
d_x_ana += d_pb @ wb[:, dm:]

# sigmoid backward for alpha
d_pal = d_al * al * (1 - al)
d_h_ana += d_pal @ wg[:, :dm]
d_x_ana += d_pal @ wg[:, dm:]

print("ref grad h:", href[0].tolist())
print("ana grad h:", d_h_ana[0].tolist())
hd = (href - d_h_ana).abs().max().item()
print(f"match: {hd < 1e-4} (diff={hd:.6f})")
