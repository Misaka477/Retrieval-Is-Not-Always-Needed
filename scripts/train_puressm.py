"""
Pure SSM baseline — RINA without attractor.
Same architecture (gate + norm + head), no patterns, no Hebbian, no attractor.
Same training config as code-seq256. Compare ppl to verify attractor contribution.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
import torch, torch.nn as nn, torch.nn.functional as F, random
from tqdm import tqdm
import numpy as np

device = "cuda"; torch.manual_seed(42); random.seed(42)
V, DM = 4096, 840
SEQ, BS = 64, 8
LR = 1e-4; SUBSAMPLE = 8; MAX_TOKENS = 200_000_000
CKPT_NAME = "pure_ssm"
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Config: dm={DM} seq={SEQ} bs={BS} lr={LR} max_tokens={MAX_TOKENS:,}")

# ── Pure SSM model (no attractor, no patterns) ──
class PureSSM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []
        for t in range(seq_len):
            combined = torch.cat([h, emb[:, t, :]], dim=-1)
            a = torch.sigmoid(self.gate_a(combined))
            b = torch.sigmoid(self.gate_b(combined))
            xp = self.proj_in(emb[:, t, :])
            h = a * h + b * xp
            h = self.norm(h)
            logits.append(self.head(self.state_norm(h)))
        return torch.stack(logits, dim=1)

# Load RINA checkpoint to initialize embedding weights
print("Initializing embedding from RINA checkpoint...")
sd = torch.load("checkpoints/code_seq256_resume.pt", map_location=device, weights_only=False)
rsd = sd["model"] if "model" in sd else sd

m = PureSSM(V, DM).to(device)
# Copy embedding and head weights from RINA
with torch.no_grad():
    m.embed.weight.copy_(rsd["embed.weight"])
    m.head.weight.copy_(rsd["head.weight"])
    m.head.bias.copy_(rsd["head.bias"])
    m.state_norm.weight.copy_(rsd["state_norm.weight"])
    m.state_norm.bias.copy_(rsd["state_norm.bias"])
    # Gate weights map: RINA's cell.gate_a/b -> our gate_a/b
    m.gate_a.weight.copy_(rsd["cell.gate_a.weight"])
    m.gate_a.bias.copy_(rsd["cell.gate_a.bias"])
    m.gate_b.weight.copy_(rsd["cell.gate_b.weight"])
    m.gate_b.bias.copy_(rsd["cell.gate_b.bias"])
    m.proj_in.weight.copy_(rsd["cell.proj_in.weight"])
    m.proj_in.bias.copy_(rsd["cell.proj_in.bias"])
    m.norm.weight.copy_(rsd["cell.norm.weight"])
    m.norm.bias.copy_(rsd["cell.norm.bias"])
print(f"  params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

# Load StarCoder data (same as code-seq256)
CODE_CACHE = os.path.join(CKPT_DIR, "code_tokens128.pt")
if os.path.exists(CODE_CACHE):
    print("Loading cached code tokens...")
    ids = torch.load(CODE_CACHE, map_location="cpu", weights_only=True)
    print(f"  tokens: {len(ids):,}")
else:
    print("Loading StarCoder...")
    ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=42)
    ids = np.empty(MAX_TOKENS, dtype=np.int32); pos = 0
    for sample in tqdm(ds, desc="tokenizing"):
        text = sample["content"]
        if len(text) < 200: continue
        token_ids = tok.encode(text).ids
        if len(token_ids) < SEQ: continue
        end = min(len(token_ids), SEQ * 1000)
        take = min(end, MAX_TOKENS - pos)
        ids[pos:pos+take] = token_ids[:take]; pos += take
        if pos >= MAX_TOKENS: break
    ids = ids[:pos]; ids = torch.tensor(ids, dtype=torch.long)
    torch.save(ids, CODE_CACHE); import gc; gc.collect()
    print(f"  tokens: {len(ids):,} (cached)")

num_batches = (len(ids) - 1) // (BS * SEQ)
train_its = num_batches // SUBSAMPLE
print(f"  batches/epoch: {train_its}")

opt = torch.optim.AdamW(m.parameters(), lr=LR)
torch.cuda.empty_cache()

for ep in range(1, 3):  # 2 epochs
    m.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/2")
    for bi in pbar:
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        logits = m(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        if bi % 200 == 0 and bi > 0:
            pbar.set_postfix(ppl=f"{torch.exp(torch.tensor(total_loss/bi)):.1f}")
    ppl = torch.exp(torch.tensor(total_loss / train_its)).item()
    print(f"ep {ep}: ppl={ppl:.1f}")
    torch.save(m.state_dict(), os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))

print(f"\nDone. Pure SSM code ppl={ppl:.1f}")
print(f"RINA (full, seq=64, code-trained): 6.60")
