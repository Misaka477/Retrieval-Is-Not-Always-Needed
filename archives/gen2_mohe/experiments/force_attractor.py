"""
Force attractor always active (th=-1) vs normal (th=1.0).
Checks if attractor has been starved by the SSM gate.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn.functional as F
from rina import TemporalSNNModel

device = "cuda"; SEQ, BS = 64, 8

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:500]
ids = torch.cat([torch.tensor(tok.encode(t).ids[:640], dtype=torch.long) for t in texts if len(tok.encode(t).ids) >= 64])
nb = max(1, (len(ids) - 1) // (BS * SEQ))

sd = torch.load("checkpoints/cann_snn15m_v2_slot_ep13.pt", map_location=device, weights_only=False)

for th, label in [(1.0, "normal"), (-1.0, "forced")]:
    m = TemporalSNNModel(4096, d_model=840, n_patterns=4096, beta=0.5,
                         attract_every=2, error_threshold=th,
                         hebbian_lr=0.0).to(device)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    m.cell.error_threshold.fill_(th)
    m.cell.att_calls.zero_()
    m.cell.total_steps.zero_()
    m.eval()

    tl = 0.0
    with torch.no_grad():
        for bi in range(nb):
            x = ids[bi * BS * SEQ : bi * BS * SEQ + BS * SEQ].view(BS, SEQ).to(device)
            logits = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 4096), x[:, 1:].reshape(-1))
            tl += loss.item()

    ppl = torch.exp(torch.tensor(tl / nb)).item()
    att = m.get_att_rate()
    print(f"{label:8s}: ppl={ppl:.1f}  att={att*100:.0f}%  calls={m.cell.att_calls.item():.0f}/{m.cell.total_steps.item():.0f}")
    # Also check the actual buffer value
    print(f"         th_actual={m.cell.error_threshold.item():.1f}")
