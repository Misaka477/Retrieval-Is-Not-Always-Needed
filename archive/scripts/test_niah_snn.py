"""NIAH + slot — 不用 buffer, Python dict build injection tensor each forward."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import torch, torch.nn.functional as F, time

device = "cuda"
dm = 840; V = 4096

from modules.temporal_snn_cell import TemporalSNNCell, TemporalSNNModel
CKPT = "checkpoints/cann_snn15m_v2_ep12.pt"
print("Loading...", flush=True)
ckpt = torch.load(CKPT, map_location=device)

# Build model without slot — inject externally
model = TemporalSNNModel(V, d_model=dm, n_patterns=4096, beta=0.5, attract_every=2,
                          error_threshold=1.0, hebbian_lr=0.0, inhibition_threshold=0.0).to(device).eval()
model.load_state_dict(ckpt["model"], strict=False)

# Fresh slot_proj (not from checkpoint)
model.slot_proj = torch.nn.Linear(dm, dm).to(device)
model.slot_proj.weight.data.zero_(); model.slot_proj.bias.data.zero_()
opt = torch.optim.AdamW(model.slot_proj.parameters(), lr=1e-3)
print(f"  slot_proj params: {sum(p.numel() for p in model.slot_proj.parameters()):,}", flush=True)

def make_niah(n, gap, n_keys=10):
    f = 2*n_keys + 1
    xl, yl = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys+1, (1,)).item()
        v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
        xl.append([k, v] + [f]*gap + [k]); yl.append(v)
    return torch.tensor(xl), torch.tensor(yl)

def forward_with_slot(model, x, slot_dict):
    """Forward with external slot injection — no buffers, no graph issues."""
    bsz, sl = x.shape; dm = model.d_model
    emb = model.embed(x); h = torch.zeros(bsz, dm, device=x.device)
    for t in range(sl):
        if t < sl - 1:
            h = model.cell(h, emb[:, t, :], step=t)
        else:
            # Build injection from dict
            inj = torch.stack([slot_dict.get(x[b,-1].item(), torch.zeros(dm,device=device)) for b in range(bsz)])
            h = model.cell(h + inj, emb[:, t, :], step=t)
            # Last-step full attractor
            pat = model.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h.unsqueeze(1)
            scores = xi @ pat.transpose(1, 2) * model.cell.beta_t[0]
            attracted = torch.softmax(scores, dim=-1) @ pat
            attracted = attracted.squeeze(1)
            combined_last = torch.cat([h, emb[:, -1, :]], dim=-1)
            alpha = torch.sigmoid(model.cell.gate_alpha(combined_last))
            h = h + alpha * (attracted - h)
            h = model.cell.norm(h)
        h_out = model.head(model.state_norm(h))
        logits = [h_out] if t == sl-1 else (logits+[h_out] if t > 0 else [h_out])
    return torch.stack(logits, dim=1)

# Train
print("\nTraining slot_proj (gap=4, 100 steps)...", flush=True)
td = make_niah(400, 4)
t0 = time.time()
for step in range(100):
    idx = torch.randint(0, len(td[0]), (32,))
    x_b, y_b = td[0][idx].to(device), td[1][idx].to(device)
    
    # Build slot dict from batch data
    slot = {}
    for b in range(32):
        k, v = int(x_b[b, 0]), int(y_b[b])
        if k > 0:
            slot[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)
    
    opt.zero_grad()
    logits = forward_with_slot(model, x_b, slot)
    loss = F.cross_entropy(logits[:, -1], y_b)
    loss.backward(); opt.step()
    if step % 20 == 19:
        print(f"  step {step+1}: loss={loss.item():.3f}", flush=True)
print(f"  done in {time.time()-t0:.0f}s", flush=True)

# Test
print(f"\nNIAH + Slot (external dict injection)")
print(f"{'='*50}")
for gap in [8, 16, 32, 64, 128, 256]:
    td_test = make_niah(50, gap)
    tx, ty = td_test[0].to(device), td_test[1].to(device)
    slot = {}
    for b in range(50):
        k, v = int(tx[b, 0]), int(ty[b])
        if k > 0:
            slot[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)
    with torch.no_grad():
        logits = forward_with_slot(model, tx, slot)
    pred = logits[:, -1].argmax(-1)
    acc = (pred == ty).float().mean().item()
    seq_len = 2+gap+1
    note = f" (seq={seq_len}>64)" if seq_len > 64 else ""
    print(f"gap={gap:4d}: {acc*100:5.1f}%{note}")
