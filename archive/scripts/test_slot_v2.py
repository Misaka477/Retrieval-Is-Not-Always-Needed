"""
Self-contained reproduction of Hopfield + embedding slot injection.
"""
import torch, torch.nn.functional as F, sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "references", "hopfield-layers"))
from hflayers import Hopfield
device = "cuda"


class RINASlotModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.hopfield = Hopfield(input_size=d_model, hidden_size=d_model,
            output_size=d_model, num_heads=1, scaling=0.5,
            update_steps_max=3, batch_first=True)
        self.slot_vals = {}
        self.slot_proj = torch.nn.Linear(d_model, d_model)
        self.head_aug = torch.nn.Linear(d_model * 2, d_model)
        self.head = torch.nn.Linear(d_model, vocab_size)

    def store(self, key_id, val_id):
        with torch.no_grad():
            ve = self.embed(torch.tensor([val_id], device=device))
            self.slot_vals[key_id] = ve.detach().cpu()

    def forward(self, x):
        bsz = x.shape[0]
        out = self.hopfield(self.embed(x))
        tid = x[0, -1].item()
        if tid in self.slot_vals:
            ve = self.slot_vals[tid].to(device)
            proj = self.slot_proj(ve).unsqueeze(0).expand(bsz, -1, -1)
            cat = torch.cat([out[:, -1:], proj], dim=-1)
            return self.head(self.head_aug(cat))
        return self.head(out)


def make_data(n, gap=8, n_keys=10):
    f = 2 * n_keys + 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys+1, (1,)).item()
        v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)


for gap in [8, 16, 32, 64, 128]:
    results = []
    for seed in [0, 42, 99, 123, 456]:
        torch.manual_seed(seed); random.seed(seed)
        xi, yi = make_data(1000, gap)
        xt, yt = make_data(200, gap)

        model = RINASlotModel(23, d_model=64).to(device)
        for b in range(1000):
            model.store(xi[b, 0].item(), xi[b, 1].item())

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best = 0
        for ep in range(60):
            model.train()
            for i in range(0, 1000, 32):
                loss = F.cross_entropy(
                    model(xi[i:i+32].to(device))[:, -1], yi[i:i+32].to(device))
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                acc = (model(xt.to(device))[:, -1].argmax(dim=-1) == yt.to(device)
                       ).float().mean().item()
                best = max(best, acc)
        results.append(best)
        print(f"  seed={seed:3d}: best={best*100:.0f}%")

    avg = sum(results) / len(results) * 100
    print(f"gap={gap:3d}: avg={avg:.0f}% (best={max(results)*100:.0f}%, worst={min(results)*100:.0f}%)")
    print()
