"""Generation stability test -- manifold integrity over 2048 steps."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm

device = "cuda"
from rina import TemporalSNNModel
ckpt = torch.load("checkpoints/cann_snn15m_v2_ep12.pt", map_location=device, weights_only=False)
m = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5,
                     attract_every=2, error_threshold=1.0,
                     hebbian_lr=0.0, inhibition_threshold=0.0).to(device)
m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
m.eval()
patterns = m.cell.patterns.detach()

MAX_LEN = 2048
x = torch.randint(0, 4096, (1, 64)).to(device)
state = torch.zeros(1, 840, device=device)
cos_trace = []; tokens = []

with torch.no_grad():
    for t in tqdm(range(MAX_LEN)):
        emb = m.embed(x[:, -1:])
        state = m.cell(state, emb[:, 0, :], step=t)
        logit = m.head(m.state_norm(state))
        next_id = logit.argmax(dim=-1)
        tokens.append(next_id.item())
        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)[:, -64:]
        state_n = state / state.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pat_n = patterns / patterns.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_trace.append((state_n @ pat_n.T).max().item())

cos_trace = torch.tensor(cos_trace)
repeats = sum(1 for i in range(100, len(tokens)) if tokens[i] == tokens[i-1])

print(f"\n{'='*50}")
print(f"Generation stability report (2048 steps)")
print(f"{'='*50}")
print(f"State-basin max cosine:")
print(f"  start-10:  {cos_trace[:10].mean():.3f}")
print(f"  mid-1024:  {cos_trace[1024:1034].mean():.3f}")
print(f"  end-100:   {cos_trace[-100:].mean():.3f}")
print(f"  min:       {cos_trace.min():.3f}")
print(f"  var:       {cos_trace.var():.4f}")
print(f"Tokens: {len(set(tokens))}/{4096} unique ({100*len(set(tokens))/4096:.1f}%)")
print(f"Repeats (last 1024): {repeats}")

if cos_trace[-100:].mean() > 0.9 and cos_trace.var() < 0.01:
    print(f"-> MANIFOLD STABLE: attractor maintains state within basin.")
    print(f"-> Strong evidence for contraction guarantee over 2048 steps.")
elif cos_trace[-100:].mean() > 0.7:
    print(f"-> MANIFOLD WEAK: still attracted but drifted from basin peak.")
else:
    print(f"-> MANIFOLD BROKEN: state not sustained near any basin.")
    print(f"-> Root cause: seq=64 training does not expose gate to 2048-step dynamics.")
