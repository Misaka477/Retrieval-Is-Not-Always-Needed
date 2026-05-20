"""NIAH recall on trained 15M — CANN vs ablation (slot + attractor value proof)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, time
from modules.cann_ssm import RINASeqModel, _full_forward

device = "cuda"; torch.manual_seed(42)

DM, NP, V = 768, 4096, 4096
N_KEYS = 20
V_NIAH = 2 * N_KEYS + 2

def make_niah(n, gap=8):
    f = V_NIAH - 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, N_KEYS + 1, (1,)).item()
        v = torch.randint(N_KEYS + 1, 2 * N_KEYS + 1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)

def load_model(ckpt_path, attract_every):
    model = RINASeqModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                         n_slots=V_NIAH, attract_every=attract_every).to(device)
    st = torch.load(ckpt_path, map_location=device)
    st.pop("embed.weight", None); st.pop("head.weight", None); st.pop("head.bias", None)
    st.pop("state_norm.weight", None); st.pop("state_norm.bias", None)
    model.load_state_dict(st, strict=False)
    model.slot_table.zero_()
    return model

print("Loading models...", flush=True)
cann = load_model("checkpoints/cann_15m_wt103_final.pt", attract_every=2)
abl  = load_model("checkpoints/cann_15m_abl_final.pt", attract_every=9999)
print("  Done.", flush=True)

def test_model(model, name, gap, train_x, train_y, test_x, test_y, steps=200):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0
    B = 32  # mini-batch
    for step in range(steps):
        model.zero_grad()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), B):
            idx = perm[i:i+B]
            logits = _full_forward(
                train_x[idx].to(device), model.embed.weight, model.slot_table,
                model.head.weight, model.head.bias,
                model.state_norm.weight, model.state_norm.bias,
                model.cell.patterns, model.cell.beta_t,
                model.cell.gate_a.weight, model.cell.gate_a.bias,
                model.cell.gate_b.weight, model.cell.gate_b.bias,
                model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                model.cell.proj_in.weight, model.cell.proj_in.bias,
                model.cell.norm.weight, model.cell.norm.bias,
                model.attract_every,
            )
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(device))
            loss.backward()
        opt.step()

        with torch.no_grad():
            for b in range(train_x.shape[0]):
                k, v = int(train_x[b, 0]), int(train_y[b])
                if k > 0 and v > 0: model.slot_write(k, v)

        if step % 10 == 9:
            model.eval()
            with torch.no_grad():
                lt = _full_forward(
                    test_x.to(device), model.embed.weight, model.slot_table,
                    model.head.weight, model.head.bias,
                    model.state_norm.weight, model.state_norm.bias,
                    model.cell.patterns, model.cell.beta_t,
                    model.cell.gate_a.weight, model.cell.gate_a.bias,
                    model.cell.gate_b.weight, model.cell.gate_b.bias,
                    model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                    model.cell.proj_in.weight, model.cell.proj_in.bias,
                    model.cell.norm.weight, model.cell.norm.bias,
                    model.attract_every,
                )
            acc = (lt[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
            best = max(best, acc)
            print(f"  {name} gap={gap:3d} step={step+1:3d}: acc={acc*100:.0f}% best={best*100:.0f}%")
            if best >= 1.0: break
            model.train()
    return best

print("\n  gap    CANN+slot    ABL+slot    delta")
print("  ─────────────────────────────────────")
for gap in [8, 16, 32, 64, 128]:
    train_x, train_y = make_niah(400, gap)
    test_x, test_y = make_niah(200, gap)
    bc = test_model(cann, "CANN", gap, train_x, train_y, test_x, test_y)
    ba = test_model(abl,  "ABL ", gap, train_x, train_y, test_x, test_y)
    print(f"  {gap:3d}     {bc*100:.0f}%         {ba*100:.0f}%         {bc-ba:+.0f}%")
print()
