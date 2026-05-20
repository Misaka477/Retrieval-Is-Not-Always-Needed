"""NIAH slot training v2 — key fix: full seq as input so last token = key."""
import torch
import torch.nn.functional as F
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel, _full_forward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def make_niah(n, gap=8, n_keys=10):
    """Full seq input: [key, value, filler..., key], target = value (scalar)."""
    f = 2 * n_keys + 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys + 1, (1,)).item()
        v = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)

V = 2 * 10 + 2
K = 10

for gap in [8, 16, 32, 64, 128]:
    t0 = time.time()
    train_x, train_y = make_niah(800, gap, n_keys=K)
    test_x, test_y = make_niah(200, gap, n_keys=K)

    model = RINASeqModel(V, d_model=64, n_patterns=1024, beta=0.5,
                         n_slots=V, attract_every=4).to(device)
    model.slot_table.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for ep in range(80):
        model.zero_grad()
        logits = _full_forward(
            train_x.to(device), model.embed.weight, model.slot_table,
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
        loss = F.cross_entropy(logits[:, -1], train_y.to(device))
        loss.backward()
        opt.step()

        # Force-write slot at position 0: (key_id=first_token → value_id)
        with torch.no_grad():
            for b in range(train_x.shape[0]):
                k = int(train_x[b, 0])
                v = int(train_y[b])
                if k > 0 and v > 0:
                    model.slot_write(k, v)

        model.eval()
        with torch.no_grad():
            logits_test = _full_forward(
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
            pred = logits_test[:, -1].argmax(dim=-1)
            acc = (pred == test_y.to(device)).float().mean().item()
            best = max(best, acc)

        if ep % 20 == 19 or (best > 0.5 and ep < 5):
            print(f"  gap={gap:3d} ep={ep:2d}: acc={acc*100:.0f}% best={best*100:.0f}%")

    print(f"gap={gap:3d}: best={best*100:.0f}% ({(time.time()-t0)/60:.1f}min)")
    print()
