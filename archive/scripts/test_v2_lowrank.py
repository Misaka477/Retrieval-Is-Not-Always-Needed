"""Quick test: low-rank patterns v2 vs v1 — speed + ppl check."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from tokenizers import Tokenizer; from datasets import load_dataset
import torch, torch.nn.functional as F, time
from modules.cann_ssm import RINASeqModel, _full_forward

device = "cuda"; torch.manual_seed(42)
V, DM, BS, SEQ = 4096, 768, 8, 64

ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
raw_text = [t for t in ds["text"] if len(t) > 100][:50]
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
ids = torch.cat([torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in raw_text if len(tok.encode(t).ids) > SEQ])
x = ids[:SEQ].unsqueeze(0).expand(BS, -1).to(device)
y = ids[1:SEQ+1].unsqueeze(0).expand(BS, -1).to(device)
print(f"Testing with {len(ids):,} tokens (offline cache)", flush=True)

print("Creating v1 (full-rank, np=4096)...", flush=True)
m1 = RINASeqModel(V, d_model=DM, n_patterns=4096, beta=0.5, attract_every=2).to(device)
n1 = sum(p.numel() for p in m1.parameters()); print(f"  params: {n1:,}", flush=True)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.float16):
        out1 = _full_forward(x, m1.embed.weight, m1.slot_table,
            m1.head.weight, m1.head.bias, m1.state_norm.weight, m1.state_norm.bias,
            m1.cell.effective_patterns, m1.cell.beta_t,
            m1.cell.gate_a.weight, m1.cell.gate_a.bias,
            m1.cell.gate_b.weight, m1.cell.gate_b.bias,
            m1.cell.gate_alpha.weight, m1.cell.gate_alpha.bias,
            m1.cell.proj_in.weight, m1.cell.proj_in.bias,
            m1.cell.norm.weight, m1.cell.norm.bias, m1.attract_every)
    torch.cuda.synchronize()
t1 = (time.time() - t0) / 5 * 1000; loss1 = F.cross_entropy(out1.reshape(-1, V), y.reshape(-1)).item()
print(f"v1: {t1:.1f}ms/fwd  loss={loss1:.3f}", flush=True)

for r in [256, 128, 64]:
    print(f"Creating v2 (low-rank, np=4096, r={r})...", flush=True)
    m2 = RINASeqModel(V, d_model=DM, n_patterns=4096, beta=0.5, attract_every=2, pattern_rank=r).to(device)
    n2 = sum(p.numel() for p in m2.parameters()); print(f"  params: {n2:,}", flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.float16):
            out2 = _full_forward(x, m2.embed.weight, m2.slot_table,
                m2.head.weight, m2.head.bias, m2.state_norm.weight, m2.state_norm.bias,
                m2.cell.effective_patterns, m2.cell.beta_t,
                m2.cell.gate_a.weight, m2.cell.gate_a.bias,
                m2.cell.gate_b.weight, m2.cell.gate_b.bias,
                m2.cell.gate_alpha.weight, m2.cell.gate_alpha.bias,
                m2.cell.proj_in.weight, m2.cell.proj_in.bias,
                m2.cell.norm.weight, m2.cell.norm.bias, m2.attract_every)
        torch.cuda.synchronize()
    t2 = (time.time() - t0) / 5 * 1000; loss2 = F.cross_entropy(out2.reshape(-1, V), y.reshape(-1)).item()
    print(f"v2 r={r:3d}: {t2:.1f}ms/fwd  loss={loss2:.3f}  speedup={t1/t2:.1f}x  Δloss={loss2-loss1:+.3f}", flush=True)

# ── v3 MIMO 8 heads ──
print("Creating v3 (MIMO 8 heads, full-rank, np=4096/8=512)...", flush=True)
m3 = RINASeqModel(V, d_model=DM, n_patterns=4096, beta=0.5, attract_every=2, n_heads=8).to(device)
n3 = sum(p.numel() for p in m3.parameters()); print(f"  params: {n3:,}", flush=True)
# Step through tokens using cell.forward directly (not JIT _full_forward)
def step_fwd(model, x_seq):
    h = torch.zeros(BS, DM, device=device)
    emb = model.embed(x_seq)
    for t in range(SEQ):
        h = model.cell(h, emb[:, t], step=t)
    return model.head(model.state_norm(h))
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.float16):
        out3 = step_fwd(m3, x)
    torch.cuda.synchronize()
t3 = (time.time() - t0) / 5 * 1000
print(f"v3 MIMO 8: {t3:.1f}ms/fwd  speedup vs v1={t1/t3:.1f}x", flush=True)
