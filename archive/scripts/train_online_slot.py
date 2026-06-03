"""
Phase 2: Train CANN-SSM with online prediction-error-driven slot writes.

Key differences from Phase 1:
  - Slot starts EMPTY (no pre-filled data)
  - After each batch, model writes high-error predictions to slot
  - Future batches benefit from stored slot entries
  - Tests if the model can bootstrap its own memory
"""
import torch, torch.nn.functional as F, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.cann_ssm import RINASeqModel


def make_data(n, gap=8, n_keys=10):
    f = 2 * n_keys + 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys+1, (1,)).item()
        v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

V, K = 23, 10

def make_shifted(n, gap=8, n_keys=10):
    """Return (inputs, next_token_targets) for language modeling."""
    f = 2 * n_keys + 1
    seq_len = gap + 3
    inputs, targets = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys+1, (1,)).item()
        v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
        seq = [k, v] + [f] * gap + [k]
        # next-token prediction targets: what comes after each position
        inp = seq[:-1]
        tgt = seq[1:]  # predict the next token at each position
        inputs.append(inp)
        targets.append(tgt)
    return torch.tensor(inputs), torch.tensor(targets)

def make_niah(n, gap=8, n_keys=10):
    """Return (inputs, last_token_target) for NIAH recall (eval only)."""
    f = 2 * n_keys + 1
    x, y = [], []
    for _ in range(n):
        k = torch.randint(1, n_keys+1, (1,)).item()
        v = torch.randint(n_keys+1, 2*n_keys+1, (1,)).item()
        x.append([k, v] + [f] * gap + [k])
        y.append(v)
    return torch.tensor(x), torch.tensor(y)


for gap in [8, 16]:
    t0 = time.time()
    xi, yi = make_shifted(600, gap)       # next-token prediction targets (seq_len-1)
    xt, yt = make_niah(100, gap)          # NIAH eval (full seq, last token target)

    model = RINASeqModel(V, d_model=64, n_patterns=1024, beta=0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best = 0

    for ep in range(60):
        model.train()
        for i in range(0, 600, 16):
            x = xi[i:i+16].to(device)
            y = yi[i:i+16].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            per_step = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), reduction='none'
            ).reshape(x.shape[0], -1)
            model.learn_from_errors(x, y, per_step.detach().cpu(), threshold=0.3)

        model.eval()
        with torch.no_grad():
            logits = model(xt.to(device))
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == yt.to(device)).float().mean().item()
            best = max(best, acc)

        if ep % 10 == 9 or (model.slot.size > 0 and ep < 5):
            print(f"  gap={gap:3d} ep={ep:2d}: acc={acc*100:.0f}% best={best*100:.0f}% slot={model.slot.size}")

    print(f"gap={gap:3d}: best={best*100:.0f}% slot={model.slot.size} ({(time.time()-t0)/60:.1f}min)\n")
