"""
Code seq=256 progressive — load from seq=128 checkpoint.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import signal
def _clean_exit(sig, frame):
    print("\nSIGINT received — synchronizing CUDA before exit...", flush=True)
    import torch
    torch.cuda.synchronize(); torch.cuda.empty_cache(); sys.exit(0)
signal.signal(signal.SIGINT, _clean_exit)

from tokenizers import Tokenizer; from datasets import load_dataset
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel; from rina.drift import DriftTracker
from tqdm import tqdm; import numpy as np

device = "cuda"; torch.manual_seed(42); random.seed(42)
V, DM, NP, AE = 4096, 840, 4096, 2
SEQ, BS = 256, 2
LR = 1e-4; EPOCHS = 1; SUBSAMPLE = 8
MAX_TOKENS = 200_000_000; CKPT_NAME = "code_seq256"
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
RESUME_PATH = os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt")
tok = Tokenizer.from_file("checkpoints/cann_15m_wt103-vocab.json")

print(f"Config: dm={DM} seq={SEQ} bs={BS} lr={LR} ep={EPOCHS} max_tokens={MAX_TOKENS:,}")

src = "checkpoints/code_seq128_resume.pt"
print(f"Loading seq128 checkpoint: {src}", flush=True)
sd = torch.load(src, map_location=device, weights_only=False)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.8, n_slots=V).to(device)
model.load_state_dict(sd["model"], strict=False)
model.cell.hebbian_decay = 1.0; model.cell.hebbian_lr = 0.001
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)
drift_tracker = DriftTracker(compute_coverage=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler()

CODE_CACHE = os.path.join(CKPT_DIR, "code_tokens256.pt")
if os.path.exists(CODE_CACHE):
    print("Loading cached tokens...", flush=True)
    ids = torch.load(CODE_CACHE, map_location="cpu", weights_only=True)
    print(f"  tokens: {len(ids):,}", flush=True)
else:
    print("Loading StarCoder code...", flush=True)
    ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=42)
    print(f"Tokenizing (seq={SEQ}, ~200M tokens)...", flush=True)
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
    print(f"  tokens: {len(ids):,} (cached)", flush=True)

num_batches = (len(ids) - 1) // (BS * SEQ)
train_its = num_batches // SUBSAMPLE
print(f"  batches/epoch: {train_its}", flush=True)
start_ep = 1; resume_steps = 0
if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"]); start_ep = ckpt["ep"]
    resume_steps = ckpt.get("train_steps", 0)
    print(f"  resume from ep {start_ep}, step {resume_steps}", flush=True)

torch.cuda.empty_cache()
print("Warmup (3 steps on real data)...", flush=True)
model.train()
x0 = ids[:BS * SEQ].view(BS, SEQ).to(device)
for _ in range(3):
    opt.zero_grad()
    with torch.amp.autocast("cuda"): l_w = model(x0)
    l_w = F.cross_entropy(l_w[:, :-1].reshape(-1, V), x0[:, 1:].reshape(-1))
    scaler.scale(l_w).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize(); print("  done", flush=True)

for ep in range(start_ep, EPOCHS + 1):
    model.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/{EPOCHS}")
    train_steps = resume_steps if ep == start_ep else 0
    skip_until = resume_steps if ep == start_ep else 0
    drift_anchor = model.cell.patterns.detach().clone()
    log_file = os.path.join(CKPT_DIR, f"code_seq256_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,ppl,att_pct,dead_pct,norms_mu,cos_drift,frob_drift,r95,lr\n")
    for bi in pbar:
        train_steps += 1
        if train_steps <= skip_until: continue
        if model.n_slots > 0: model.slot_table.zero_()
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total_loss += loss.item()
        if train_steps % 200 == 0:
            ppl = torch.exp(torch.tensor(total_loss / (train_steps - resume_steps))).item()
            att = model.get_att_rate()
            P = model.cell.patterns.detach(); norms = P.norm(dim=-1)
            dead = (norms < 0.01).float().mean().item(); norms_mu = norms.mean().item()
            Pn = P / norms.unsqueeze(-1).clamp(min=1e-8)
            An = drift_anchor / drift_anchor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cos_drift = (Pn * An).sum(dim=-1).mean().item()
            frob_drift = (P - drift_anchor).norm().item() / drift_anchor.norm().item()
            _, S, _ = torch.linalg.svd(P.float(), full_matrices=False)
            cumvar = (S.cumsum(0) / S.sum()).tolist()
            r95 = next((i for i, v in enumerate(cumvar) if v >= 0.95), len(S))
            pbar.set_postfix(ppl=f"{ppl:.1f}", att=f"{att*100:.0f}%", dead=f"{dead*100:.0f}%",
                           mu=f"{norms_mu:.1f}", cos=f"{cos_drift:.3f}", frob=f"{frob_drift:.3f}")
            with open(log_file, "a") as f:
                lr_now = opt.param_groups[0]["lr"]
                f.write(f"{train_steps},{ppl:.2f},{att*100:.2f},{dead*100:.2f},{norms_mu:.2f},{cos_drift:.4f},{frob_drift:.4f},{r95},{lr_now:.2e}\n"); f.flush()
            if train_steps % 2000 == 0:
                ep_ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                           "ep": ep, "ppl": ppl, "train_steps": train_steps}
                torch.save(ep_ckpt, RESUME_PATH)
    effective_steps = train_steps - resume_steps
    ppl = torch.exp(torch.tensor(total_loss / effective_steps)).item()
    drift = drift_tracker.step(model.cell.patterns)
    print(f"ep {ep}: ppl={ppl:.1f} drift: cos={drift['avg_cos']:.4f} frob={drift['frob_drift']:.4f} "
          f"norms[mu={drift['norms_mean']:.2f}] dead={drift['dead_frac']*100:.0f}% cov[r95={drift['eff_rank_95']}]")
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "ep": ep, "ppl": ppl, "train_steps": train_steps}
    torch.save(ckpt, RESUME_PATH)
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))

print(f"\nDone. seq={SEQ} code ppl={ppl:.1f}")
