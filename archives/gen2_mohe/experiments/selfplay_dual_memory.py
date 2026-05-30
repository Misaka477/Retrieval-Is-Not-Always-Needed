"""
Exp: Self-play dual-memory RINA.
Fast memory (SSM gate): efficiency, quick prediction.
Slow memory (linear field P): structural, cross-token associations.
Game: fast wants to shortcut, slow wants to reveal hidden structure.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, random

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 256, 1024
SEQ, BS = 64, 8
LR = 3e-4; EPOCHS = 5
SUBSAMPLE = 4; MAX_TOKENS = 50_000_000

class SelfPlayCell(nn.Module):
    """Dual-memory cell: fast SSM + slow linear field, gated competition."""
    def __init__(self, dm, np_):
        super().__init__()
        self.dm = dm
        # Fast memory: SSM gate
        self.gate_a = nn.Linear(dm * 2, dm)
        self.gate_b = nn.Linear(dm * 2, dm)
        self.proj_in = nn.Linear(dm, dm)
        # Slow memory gate (controls retrieval from field)
        self.slow_gate = nn.Linear(dm * 2, 1)
        # Output norms
        self.fast_norm = nn.LayerNorm(dm)
        self.slow_norm = nn.LayerNorm(dm)
        # Patterns (the field)
        self.patterns = nn.Parameter(torch.randn(np_, dm) * 0.02)
        # Field mixing: controls how much slow correction applies to h
        self.field_mix = nn.Linear(dm, dm)
        # Hebbian
        self.hebbian_lr = LR

    def forward(self, h, x_emb, step=0):
        bsz = h.shape[0]
        combined = torch.cat([h, x_emb], dim=-1)

        # ── Fast memory: SSM gate (shortcut / efficiency) ──
        a = torch.sigmoid(self.gate_a(combined))
        b = torch.sigmoid(self.gate_b(combined))
        xp = self.proj_in(x_emb)
        h_fast = a * h + b * xp
        h_fast = self.fast_norm(h_fast)

        # ── Slow memory: linear field P = patterns.T @ patterns ──
        P = self.patterns.T @ self.patterns  # [dm, dm], learned association field
        field_force = h_fast @ P               # linear projection through field
        field_force = self.field_mix(field_force)
        field_force = self.slow_norm(field_force)

        # ── Gate: slow-fast competition ──
        gate = torch.sigmoid(self.slow_gate(combined))
        h_out = h_fast + gate * field_force * 0.1

        # ── Hebbian update: fast error drives field evolution ──
        if self.training:
            with torch.no_grad():
                error = (h_fast - h_out).norm(dim=-1) / (h_out.norm(dim=-1) + 1e-8)
                k_pred = (h_out.unsqueeze(1) @ self.patterns.T.unsqueeze(0)).squeeze(1).argmax(dim=-1)
                lr = self.hebbian_lr / (1.0 + error)
                lr = lr.unsqueeze(-1)
                dh = h_out - self.patterns[k_pred]
                active = error > 0.1
                if active.any():
                    pk = k_pred[active]
                    self.patterns.data.index_add_(0, pk, lr[active] * dh[active])

        return h_out


class DualMemoryModel(nn.Module):
    def __init__(self, vocab, dm, np_):
        super().__init__()
        self.embed = nn.Embedding(vocab, dm)
        self.cell = SelfPlayCell(dm, np_)
        self.head = nn.Linear(dm, vocab)
        self.state_norm = nn.LayerNorm(dm)

    def forward(self, x):
        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.cell.dm, device=x.device)
        logits = []
        for t in range(seq_len):
            h = self.cell(h, emb[:, t, :], step=t)
            logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)


# ── Data ──
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")
print("Loading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t["text"] for t in ds if len(t["text"]) > 100][:20000]
ids_list = []
for t in tqdm(texts, desc="tokenizing"):
    e = tok.encode(t).ids[:SEQ * 20]
    if len(e) >= SEQ:
        ids_list.append(torch.tensor(e, dtype=torch.long))
ids = torch.cat(ids_list)[:MAX_TOKENS]
nb = (len(ids) - 1) // (BS * SEQ)
print(f"  {len(ids):,} tokens, {nb} batches/epoch")

# ── Model ──
model = DualMemoryModel(VOCAB, DM, NP).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Training ──
print("Training dual-memory self-play...")
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    its = nb // SUBSAMPLE
    pbar = tqdm(range(its), desc=f"ep {ep}/{EPOCHS}")
    for bi in pbar:
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        if bi % 100 == 99:
            ppl = torch.exp(torch.tensor(total_loss / (bi + 1))).item()
            pbar.set_postfix(ppl=f"{ppl:.1f}")
    ppl = torch.exp(torch.tensor(total_loss / its)).item()
    print(f"  ep {ep}: ppl={ppl:.1f}")

# ── Generate ──
model.eval()
for prompt in ["The meaning of life is", "In the beginning"]:
    ids = tok.encode(prompt).ids[:10]
    gen = ids[:]
    for _ in range(80):
        inp = torch.tensor([gen[-SEQ:]], device=device)
        logits = model(inp)
        next_id = logits[0, -1].argmax().item()
        gen.append(next_id)
    text = tok.decode(gen).replace("\u0120", " ").replace("\u010a", "\n")
    print(f"\nPrompt: {prompt}\n{text[:200]}")
