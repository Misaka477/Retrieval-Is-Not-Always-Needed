"""
Test RINA v3: CANN + HashSlot + logit bias injection on NIAH recall.
"""
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rina_v3 import RINAModel


def test_niah(gap=8, n_train=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_keys = 10
    PAD, FILLER = 0, 2 * n_keys + 1
    vocab = 2 * n_keys + 3
    seq_len = gap + 3

    print(f"vocab={vocab}, seq_len={seq_len}, gap={gap}")

    def make(n):
        x, y = [], []
        for _ in range(n):
            k = torch.randint(1, n_keys+1, (1,)).item()
            v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
            seq = [k, v] + [FILLER] * gap + [k]
            tgt = [PAD] * (seq_len - 1) + [v]
            x.append(seq)
            y.append(tgt)
        return torch.tensor(x), torch.tensor(y)

    xi, yi = make(n_train)
    xt, yt = make(200)

    model = RINAModel(vocab_size=vocab, d_model=64, n_patterns=1024,
                      beta=0.5, n_iter=3, n_slots=2048, error_threshold=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Pre-populate slot with all key->value bindings from training data
    for b in range(n_train):
        k = xi[b, 0].item()
        v = xi[b, 1].item()
        if 1 <= k <= 10 and 11 <= v <= 20:
            model.slot.insert(k, 0, v)

    for epoch in range(30):
        model.train()
        loss_acc = 0
        for i in range(0, n_train, 16):
            x = xi[i:i+16].to(device)
            y = yi[i:i+16].to(device)
            logits = model.forward_with_slot_writes(x)
            loss = F.cross_entropy(logits[:, -1], y[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(xt.to(device))
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == yt.to(device)[:, -1]).float().mean().item()
            bar = "#" * int(acc * 20)
            if epoch % 5 == 0 or acc > 0.15:
                print(f"  epoch {epoch:2d}: loss={loss_acc/(n_train//16):.4f}, "
                      f"recall={acc*100:.0f}% {bar}")

    model.eval()
    with torch.no_grad():
        logits = model(xt.to(device))
        pred = logits[:, -1].argmax(dim=-1)
        acc = (pred == yt.to(device)[:, -1]).float().mean().item()
    print(f"\nFinal: gap={gap}, recall={acc*100:.1f}%")
    return acc


if __name__ == "__main__":
    test_niah(gap=8, n_train=400)
