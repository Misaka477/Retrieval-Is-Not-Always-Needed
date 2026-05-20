"""Verify each backward step of the analytical backward vs autograd."""
import torch

device = "cuda"
torch.manual_seed(0)
dm, np_ = 4, 4

h = torch.randn(1, dm, device=device, requires_grad=True)
x = torch.randn(1, dm, device=device, requires_grad=True)
p = torch.randn(np_, dm, device=device)

def lin(od, id_):
    return torch.randn(od, id_, device=device), torch.randn(od, device=device)

wa, ba = lin(dm, dm*2); wb, bb = lin(dm, dm*2)
wg, bg = lin(dm, dm*2); wp, bp = lin(dm, dm)
wn = torch.randn(dm, device=device); bn = torch.randn(dm, device=device)
beta = 0.5
N = dm

# Full PyTorch reference gradient
h2 = h.detach().requires_grad_(True)
x2 = x.detach().requires_grad_(True)
c2 = torch.cat([h2, x2], dim=-1)
a2 = torch.sigmoid(c2 @ wa.T + ba)
b2 = torch.sigmoid(c2 @ wb.T + bb)
hs2 = a2 * h2 + b2 * (x2 @ wp.T + bp)
sc2 = (hs2 @ p.T) * beta
an2 = torch.softmax(sc2, dim=-1)
al2 = torch.sigmoid(c2 @ wg.T + bg)
hn2 = hs2 + al2 * (an2 @ p - hs2)
iv2 = torch.rsqrt(hn2.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
mn2 = hn2.mean(dim=-1, keepdim=True)
ho2 = wn * (hn2 - mn2) * iv2 + bn
ho2.sum().backward()
ref_h = h2.grad.clone()

# Now compute analytical backward step by step
with torch.no_grad():
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
    mn = hn.mean(dim=-1, keepdim=True)
    hnorm = (hn - mn) * iv

# Step 1: LayerNorm backward
d_out = torch.ones_like(hn)  # gradient from sum of output
d_hn = d_out * wn  # d(y*w+b) = d_out * w
m = d_hn.mean(dim=-1, keepdim=True)
s = (d_hn * hnorm).mean(dim=-1, keepdim=True)
d_h_new_ana = (d_hn - m - hnorm * s) * iv
print(f"LayerNorm backward: d_h_new_ana = {d_h_new_ana[0].tolist()}")

# Step 2: h_new = hssm + al * (at - hssm)
diff = at - hssm
d_al_ana = (d_h_new_ana * diff).sum(dim=-1, keepdim=True)
d_hssm_ana = d_h_new_ana * (1 - al) 
d_at_ana = d_h_new_ana * al
print(f"d_al_ana = {d_al_ana.item():.6f}")

# Step 3: at = an @ p → attracted
d_an_ana = d_at_ana @ p.T  # (1, dm) @ (dm, np) = (1, np)
print(f"d_an_ana = {d_an_ana[0].tolist()}")

# Step 4: softmax backward
# d_softmax = an * (d_an - sum(an * d_an))
d_an_corrected = an * (d_an_ana - (an * d_an_ana).sum(dim=-1, keepdim=True))
print(f"d_an_corrected = {d_an_corrected[0].tolist()}")

# Step 5: scores = (hssm @ p.T) * beta
d_sc_ana = d_an_corrected * beta
print(f"d_sc_ana = {d_sc_ana[0].tolist()}")

# Step 6: h_ssm from attractor
d_hssm_attn = d_sc_ana @ p  # (1, np) @ (np, dm) = (1, dm)
d_hssm_ana = d_hssm_ana + d_hssm_attn
print(f"d_hssm (after attn) = {d_hssm_ana[0].tolist()}")

# Step 7: h_ssm = a * h + b * xp
d_a_ana = d_hssm_ana * h
d_b_ana = d_hssm_ana * xp
d_xp_ana = d_hssm_ana * b
d_h_ana = d_hssm_ana * a
print(f"d_h (from h_ssm) = {d_h_ana[0].tolist()}")

# Step 8: xp = x @ wp.T + bp
d_x_ana = d_xp_ana @ wp
print(f"d_x (from xp)   = {d_x_ana[0].tolist()}")

# Step 9: Sigmoid backward for a
d_pa = d_a_ana * a * (1 - a)
d_h_ana = d_h_ana + d_pa @ wa[:, :N]
d_x_ana = d_x_ana + d_pa @ wa[:, N:]
print(f"d_h (after wa)   = {d_h_ana[0].tolist()}")
print(f"d_x (after wa)   = {d_x_ana[0].tolist()}")

# Step 10: Sigmoid backward for b
d_pb = d_b_ana * b * (1 - b)
d_h_ana = d_h_ana + d_pb @ wb[:, :N]
d_x_ana = d_x_ana + d_pb @ wb[:, N:]
print(f"d_h (after wb)   = {d_h_ana[0].tolist()}")
print(f"d_x (after wb)   = {d_x_ana[0].tolist()}")

# Step 11: Sigmoid backward for alpha
d_pal = d_al_ana * al * (1 - al)
d_h_ana = d_h_ana + d_pal @ wg[:, :N]
d_x_ana = d_x_ana + d_pal @ wg[:, N:]
print(f"d_h (final)       = {d_h_ana[0].tolist()}")
print(f"d_x (final)       = {d_x_ana[0].tolist()}")

print(f"\nh ref: {ref_h[0].tolist()}")
hd = (ref_h - d_h_ana).abs().max().item()
print(f"h match: {hd < 1e-4} (diff={hd:.6f})")
