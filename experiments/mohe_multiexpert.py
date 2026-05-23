"""
MoHE v3: Mixture of Hebbian Experts with winner-take-all competition.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, random
from rina.mohe import MoHE

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM = 50257, 256
SEQ, BS = 64, 8
LR = 1e-4; EPOCHS = 12
SUBSAMPLE = 4; MAX_TOKENS = 50_000_000
MAX_DEPTH = 2
CONV_THRESH = 0.05
INHIBIT_LR = 0.1
CKPT_DIR = "../checkpoints"
CKPT_NAME = "mohe_run1"
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR), exist_ok=True)
LOG_HEADER = "epoch,step,ppl,loss,lr"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}.csv")

# ── Data ──
tok = Tokenizer.from_pretrained("gpt2")
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
model = MoHE(VOCAB, DM, 512, n_experts=4).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")
for i, expert in enumerate(model.experts):
    print(f"  expert {i}: {sum(p.numel() for p in expert.parameters())/1e3:.0f}K")

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / 500))

# ── Resume ──
start_ep = 1
resume_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_resume.pt")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    start_ep = ckpt["ep"]
    print(f"  Resumed from ep {start_ep}")

# ── Training ──
print("Training MoHE with Depth-of-Thought...")
global_step = 0
for ep in range(start_ep, EPOCHS + 1):
    model.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    its = nb // SUBSAMPLE
    pbar = tqdm(range(its), desc=f"ep {ep}/{EPOCHS}")
    for bi in pbar:
        global_step += 1
        start = perm[(bi * SUBSAMPLE) % len(perm)]
        x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  NaN/Inf at step {bi}, reducing LR")
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        total_loss += loss.item()
        if bi % 100 == 99:
            ppl = torch.exp(torch.tensor(total_loss / (bi + 1))).item()
            lr_now = opt.param_groups[0]['lr']
            pbar.set_postfix(ppl=f"{ppl:.1f}", lr=f"{lr_now:.2e}")
            if not os.path.exists(LOG_PATH):
                with open(LOG_PATH, "w", newline="") as f:
                    f.write(LOG_HEADER + "\r\n")
            with open(LOG_PATH, "a", newline="") as f:
                f.write(f"{ep},{global_step},{ppl:.1f},{loss.item():.2f},{lr_now:.2e}\r\n")
    ppl = torch.exp(torch.tensor(total_loss / its)).item()
    print(f"  ep {ep}: ppl={ppl:.1f}")
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep + 1, "ppl": ppl}
    torch.save(ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
    torch.save(ckpt, resume_path)

# ── Generate ──
model.eval()
for prompt in ["The meaning of life is", "def fibonacci(n):"]:
    ids = tok.encode(prompt).ids[:10]
    gen = ids[:]
    for _ in range(100):
        inp = torch.tensor([gen[-SEQ:]], device=device)
        logits = model(inp)
        gen.append(logits[0, -1].argmax().item())
    text = tok.decode(gen).replace("\u0120", " ").replace("\u010a", "\n")
    print(f"\nPrompt: {prompt}\n{text[:300]}")
