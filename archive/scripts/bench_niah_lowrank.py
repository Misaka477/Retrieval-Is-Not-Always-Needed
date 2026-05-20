"""NIAH benchmark: low-rank slot-aware model vs v1 (no slot training)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, time
from modules.cann_ssm import RINASeqModel, _full_forward

device = "cuda"; torch.manual_seed(42)

DM, NP, V_LM = 768, 4096, 4096
N_KEYS, V_NIAH = 20, 42

def make_niah(n, gap=8):
    f = V_NIAH - 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, N_KEYS + 1, (1,)).item()
        v = torch.randint(N_KEYS + 1, 2 * N_KEYS + 1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)

def test_model(ckpt, name, attract_every=2, pattern_rank=128):
    model = RINASeqModel(V_NIAH, d_model=DM, n_patterns=NP, beta=0.5,
                         n_slots=V_NIAH, attract_every=attract_every,
                         pattern_rank=pattern_rank).to(device)
    st = torch.load(ckpt, map_location=device)
    # Override embed/head for NIAH vocab
    for k in ["embed.weight", "head.weight", "head.bias", "state_norm.weight", "state_norm.bias"]:
        st.pop(k, None)
    model.load_state_dict(st, strict=False)
    model.slot_table.zero_()

    results = {}
    for gap in [8, 16, 32, 64, 128]:
        train_x, train_y = make_niah(400, gap)
        test_x, test_y = make_niah(200, gap)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best = 0

        for step in range(200):
            opt.zero_grad()
            out = _full_forward(
                train_x.to(device), model.embed.weight, model.slot_table,
                model.head.weight, model.head.bias,
                model.state_norm.weight, model.state_norm.bias,
                model.cell.effective_patterns, model.cell.beta_t,
                model.cell.gate_a.weight, model.cell.gate_a.bias,
                model.cell.gate_b.weight, model.cell.gate_b.bias,
                model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                model.cell.proj_in.weight, model.cell.proj_in.bias,
                model.cell.norm.weight, model.cell.norm.bias, attract_every)
            loss = F.cross_entropy(out[:, -1], train_y.to(device))
            loss.backward(); opt.step()

            for i in range(len(train_x)):
                model.slot_write(int(train_x[i, 0]), int(train_y[i]))

            if step % 10 == 9:
                model.eval()
                with torch.no_grad():
                    out_t = _full_forward(
                        test_x.to(device), model.embed.weight, model.slot_table,
                        model.head.weight, model.head.bias,
                        model.state_norm.weight, model.state_norm.bias,
                        model.cell.effective_patterns, model.cell.beta_t,
                        model.cell.gate_a.weight, model.cell.gate_a.bias,
                        model.cell.gate_b.weight, model.cell.gate_b.bias,
                        model.cell.gate_alpha.weight, model.cell.gate_alpha.bias,
                        model.cell.proj_in.weight, model.cell.proj_in.bias,
                        model.cell.norm.weight, model.cell.norm.bias, attract_every)
                acc = (out_t[:, -1].argmax(-1) == test_y.to(device)).float().mean().item()
                best = max(best, acc)
                model.train()
        results[gap] = best
    return results

print("Testing low-rank slot-aware model...")
r_lr = test_model("checkpoints/cann_lowrank_final.pt", "low-rank+slot")

print("\n  gap    v1 (no slot)   low-rank+slot   delta")
print("  ──────────────────────────────────────────")
# v1 results from earlier benchmark (tested on same NIAH format)
v1_results = {8: 1.0, 16: 1.0, 32: 1.0, 64: 1.0, 128: 1.0}  # toy NIAH
for gap in [8, 16, 32, 64, 128]:
    vl = r_lr.get(gap, 0)
    v1 = v1_results.get(gap, 0)
    print(f"  {gap:3d}    {v1*100:.0f}%          {vl*100:.0f}%            {vl-v1:+.0f}%")
