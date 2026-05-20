"""
Test HopfieldLayer on NIAH recall task.

Uses the well-tested Hopfield module from references/hopfield-layers.
Hopfield with iterative retrieval (update_steps_max>0) does
associative memory = each position retrieves from all other positions.
"""
import torch, torch.nn.functional as F, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references", "hopfield-layers"))
from hflayers import Hopfield
from modules.hash_slot import HashSlot


class HopfieldMemoryModel(torch.nn.Module):
    """
    One-layer model: Embedding → Hopfield (iterative) → Head.

    Hopfield stores ALL token representations and retrieves
    via iterative dynamics (multiple refinement steps).
    """
    def __init__(self, vocab_size, d_model=64, n_heads=1, beta=None, n_iter=3):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)

        scaling = beta if beta is not None else 1.0 / (d_model ** 0.5)
        self.hopfield = Hopfield(
            input_size=d_model, hidden_size=d_model, output_size=d_model,
            num_heads=n_heads, scaling=scaling,
            update_steps_max=n_iter, update_steps_eps=1e-4,
            batch_first=True,
        )
        self.head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out = self.hopfield(emb)
        logits = self.head(out)
        return logits


class HopfieldWithSlot(torch.nn.Module):
    """
    Hopfield + HashSlot + logit bias.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=1, beta=None, n_iter=3, n_slots=2048):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        scaling = beta if beta is not None else 1.0 / (d_model ** 0.5)
        self.hopfield = Hopfield(
            input_size=d_model, hidden_size=d_model, output_size=d_model,
            num_heads=n_heads, scaling=scaling,
            update_steps_max=n_iter, update_steps_eps=1e-4,
            batch_first=True,
        )
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.slot = HashSlot(capacity=n_slots)
        self.slot_bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        emb = self.embed(x)
        out = self.hopfield(emb)
        logits = self.head(out)

        for t in range(x.shape[1]):
            token_id = x[0, t].item()
            slot_val = self.slot.lookup(token_id)
            if slot_val is not None:
                logits[:, t, slot_val] += self.slot_bias[slot_val]

        return logits


def make_niah_data(n, gap=8, n_keys=10):
    PAD, FILLER = 0, 2 * n_keys + 1
    seq_len = gap + 3
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys + 1, (1,)).item()
        v = torch.randint(n_keys + 1, 2 * n_keys + 1, (1,)).item()
        seq = [k, v] + [FILLER] * gap + [k]
        tgt = [PAD] * (seq_len - 1) + [v]
        x.append(seq)
        y.append(tgt)
    return torch.tensor(x), torch.tensor(y)


def train(model, xi, yi, xt, yt, n_epochs=30, lr=3e-4, slot_write=False, n_keys=10):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_train = xi.shape[0]

    for epoch in range(n_epochs):
        model.train()
        loss_acc = 0
        for i in range(0, n_train, 32):
            x = xi[i:i+32].to(device)
            y = yi[i:i+32].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, -1], y[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item()

            if slot_write and hasattr(model, 'slot'):
                for b in range(x.shape[0]):
                    k = x[b, 0].item()
                    v = x[b, 1].item()
                    if 1 <= k <= n_keys and v > n_keys:
                        model.slot.insert(k, 0, v)
                model.slot.tick()

        model.eval()
        with torch.no_grad():
            logits = model(xt.to(device))
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == yt.to(device)[:, -1]).float().mean().item()

        if epoch % 5 == 0 or acc > 0.15:
            slot_info = f"slot={model.slot.size}" if hasattr(model, 'slot') else ""
            print(f"  epoch {epoch:2d}: loss={loss_acc/(n_train//32):.4f}, "
                  f"recall={acc*100:.0f}% {slot_info}")

    model.eval()
    with torch.no_grad():
        logits = model(xt.to(device))
        pred = logits[:, -1].argmax(dim=-1)
        acc = (pred == yt.to(device)[:, -1]).float().mean().item()
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for gap in [8, 16, 32, 64]:
        n_keys = 10
        vocab = 2 * n_keys + 3
        seq_len = gap + 3

        xi, yi = make_niah_data(800, gap=gap)
        xt, yt = make_niah_data(200, gap=gap)

        print(f"--- Hopfield (no slot) gap={gap} (vocab={vocab}, seq={seq_len}) ---")
        model = HopfieldMemoryModel(vocab, d_model=64, n_heads=1, beta=0.5, n_iter=3).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params:,}")
        acc = train(model, xi, yi, xt, yt, n_epochs=30)
        print(f"  Final: {acc*100:.1f}%\n")
