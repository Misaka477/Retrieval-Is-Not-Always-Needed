"""NIAH Slot Recall on SNN v2 — 对标 V1 bench_niah_slot, 外部 dict slot 注入."""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import torch, torch.nn.functional as F, time

device = "cuda"; torch.manual_seed(42)

from modules.temporal_snn_cell import TemporalSNNCell

DM, NP, V = 840, 4096, 4096
N_KEYS = 20
V_NIAH = 2 * N_KEYS + 2

# 加载模型 weights (不用 slot_table — 外部注入)
CKPT = "checkpoints/cann_snn15m_v2_ep12.pt"
print("Loading SNN weights...", flush=True)

# 构建一个共享 all weights 的 bare model
def build_model():
    from modules.temporal_snn_cell import TemporalSNNModel
    m = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5, attract_every=2,
                          error_threshold=1.0, hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model"], strict=False)
    return m

model = build_model()
# Trainable slot_proj (fresh init — not from checkpoint)
model.slot_proj = torch.nn.Linear(DM, DM).to(device)
model.slot_proj.weight.data.normal_(0, 0.01)
model.slot_proj.bias.data.zero_()

def make_niah(n, gap=8):
    f = V_NIAH - 1; x, y = [], []
    for _ in range(n):
        k = torch.randint(1, N_KEYS+1, (1,)).item()
        v = torch.randint(N_KEYS+1, 2*N_KEYS+1, (1,)).item()
        x.append([k, v] + [f]*gap + [k]); y.append(v)
    return torch.tensor(x), torch.tensor(y)

def forward_with_slot(model, x, slot_dict):
    bsz, sl = x.shape; dm = model.d_model
    emb = model.embed(x); h = torch.zeros(bsz, dm, device=device)
    for t in range(sl):
        if t < sl - 1:
            h = model.cell(h, emb[:, t, :], step=t)
        else:
            inj = torch.stack([slot_dict.get(x[b, -1].item(), torch.zeros(dm, device=device)) for b in range(bsz)])
            h = model.cell(h + inj, emb[:, t, :], step=t)
            pat = model.cell.patterns.unsqueeze(0).expand(bsz, -1, -1)
            xi = h.unsqueeze(1)
            sc = xi @ pat.transpose(1, 2) * model.cell.beta_t[0]
            attracted = (torch.softmax(sc, dim=-1) @ pat).squeeze(1)
            cl = torch.cat([h, emb[:, -1, :]], dim=-1)
            alpha = torch.sigmoid(model.cell.gate_alpha(cl))
            h = h + alpha * (attracted - h); h = model.cell.norm(h)
        h_out = model.head(model.state_norm(h))
        logits = [h_out] if t == sl-1 else (logits+[h_out] if t > 0 else [h_out])
    return torch.stack(logits, dim=1)

print("\n  gap    SNN+slot")
print("  ───────────────")
for gap in [8, 16, 32, 64, 128]:
    train_x, train_y = make_niah(400, gap)
    test_x, test_y = make_niah(200, gap)

    # Train ALL parameters — matching V1 bench_niah_slot protocol
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0
    B = 32

    for step in range(200):
        model.train()
        opt.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), B):
            idx = perm[i:i+B]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)

            slot = {}
            for b in range(len(idx)):
                k, v = int(xb[b, 0]), int(yb[b])
                if k > 0 and v > 0:
                    slot[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)

            logits = forward_with_slot(model, xb, slot)
            loss = F.cross_entropy(logits[:, -1], yb)
            loss.backward()
        opt.step()

        if step % 10 == 9:
            model.eval()
            with torch.no_grad():
                slot_test = {}
                for b in range(test_x.shape[0]):
                    k, v = int(test_x[b, 0]), int(test_y[b])
                    if k > 0 and v > 0:
                        slot_test[k] = model.slot_proj(model.embed(torch.tensor([v], device=device))).squeeze(0)
                lt = forward_with_slot(model, test_x.to(device), slot_test)
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best = max(best, acc)
            print(f"  gap={gap:3d} step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")
            if best >= 1.0: break
            model.train()

    print(f"  {gap:3d}     {best*100:.0f}%")
