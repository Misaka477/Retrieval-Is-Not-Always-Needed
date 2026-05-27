"""
MoHE 83M on FineWeb + StarCoder (200M tokens). 
Depth-of-Thought, winner-take-all Hebbian, GPT-2 50K vocab.
Weight tying enabled (head.weight = embed.weight).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, random, numpy as np, io
from rina.mohe import MoHE

device = "cuda"; torch.manual_seed(42); random.seed(42)
DEBUG_MEM = False  # set True to print alloc every 10 steps
VOCAB, DM, NP = 50257, 1024, 512
SEQ, BS = 128, 8
LR = 1e-4; EPOCHS = 2
SUBSAMPLE = 8; MAX_TOKENS = 200_000_000
MAX_DEPTH = 3
CKPT_DIR = "../checkpoints"; CKPT_NAME = "mohe_83m"
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR), exist_ok=True)

print("Loading GPT-2 tokenizer...")
tok = Tokenizer.from_pretrained("gpt2")

CACHE_FW = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, "mohe_fw.npy")
CACHE_SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, "mohe_sc.npy")

def np_save_chunked(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    data = buf.getvalue()
    CHUNK = 64 * 1024 * 1024
    with open(path, 'wb') as f:
        for i in range(0, len(data), CHUNK):
            f.write(data[i:i + CHUNK])

def load_or_tokenize(src, cfg, split, cache, desc, max_tokens=MAX_TOKENS):
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, cache)
    if os.path.exists(cache_path):
        ids = np.load(cache_path, mmap_mode='r')
        ids = torch.from_numpy(ids)
        print(f"  {desc}: {len(ids):,} tokens (cached)")
        return ids
    print(f"  Loading {desc}...")
    ds = load_dataset(src, cfg, split=split, streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    arr = np.empty(max_tokens, dtype=np.int32); pos = 0; batch = []
    for s in tqdm(ds, desc=f"  Tokenizing {desc}"):
        text = s.get("text", s.get("content", ""))
        if len(text) < 200: continue
        batch.append(text)
        if len(batch) >= 500:
            for enc in tok.encode_batch(batch):
                tids = enc.ids[:SEQ * 100]
                if len(tids) >= SEQ:
                    take = min(len(tids), max_tokens - pos)
                    arr[pos:pos+take] = tids[:take]; pos += take
                    if pos >= max_tokens: break
            batch = []
            if pos >= max_tokens: break
    if batch:
        for enc in tok.encode_batch(batch):
            tids = enc.ids[:SEQ * 100]
            if len(tids) >= SEQ:
                take = min(len(tids), max_tokens - pos)
                arr[pos:pos+take] = tids[:take]; pos += take
                if pos >= max_tokens: break
    ids = torch.tensor(arr[:pos], dtype=torch.long)
    np_save_chunked(cache_path, ids.numpy())
    print(f"  {desc}: {len(ids):,} tokens (cached)")
    return ids

ids_fw = load_or_tokenize("HuggingFaceFW/fineweb", "sample-10BT", "train", "mohe_fw.npy", "FineWeb")
ids_sc = load_or_tokenize("bigcode/starcoderdata", None, "train", "mohe_sc.npy", "StarCoder")
ids_math = load_or_tokenize("open-web-math/open-web-math", None, "train", "mohe_math.npy", "OpenWebMath")
n_fw = int(MAX_TOKENS * 0.70)
n_sc = int(MAX_TOKENS * 0.15)
n_math = MAX_TOKENS - n_fw - n_sc
ids_fw = ids_fw[:n_fw]; ids_sc = ids_sc[:n_sc]; ids_math = ids_math[:n_math]
ids = torch.cat([ids_fw, ids_sc, ids_math])
perm = torch.randperm(len(ids))
ids = ids[perm]
nb = (len(ids) - 1) // (BS * SEQ)
print(f"  total: {len(ids):,} tokens, {nb} batches/epoch")

model = MoHE(VOCAB, DM, NP, n_experts=4,
             aux_loss_weight=0.5, route_noise=0.2, expert_dropout=0.2, topk=2).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")
head_lr = LR * 3
head_param_names = {id(p) for p in model.head.parameters()}
other_params = [p for n, p in model.named_parameters() if id(p) not in head_param_names]
head_params = list(model.head.parameters())
opt = torch.optim.AdamW([
    {"params": other_params, "lr": LR},
    {"params": head_params, "lr": head_lr},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / 500))

start_ep = 1
resume_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_resume.pt")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("prev_route"):
            del sd[k]
    rw = sd.get("router.weight")
    if rw is not None and rw.shape[-1] == DM * 2:
        sd["router.weight"] = rw[:, DM:]
    model.load_state_dict(sd, strict=False)
    try:
        opt.load_state_dict(ckpt["opt"])
    except ValueError:
        print("  Optimizer state incompatible (new params), reinitializing")
    start_ep = ckpt["ep"]
    if "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    print(f"  Resumed from ep {start_ep}")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}.csv")
print("Training MoHE 83M on FineWeb+StarCoder+OpenWebMath...")
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
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), label_smoothing=0.1)
        loss = loss + getattr(model, '_last_aux_loss', 0.0)
        if torch.isnan(loss) or torch.isinf(loss):
            scheduler.step()
            continue
        loss.backward()
        model.finish_training_step()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step(); scheduler.step()
        # SGDR: reset optimizer + warmup every 2000 steps
        if global_step % 2000 == 0:
            opt = torch.optim.AdamW([
                {"params": other_params, "lr": LR * 0.1},
                {"params": head_params, "lr": head_lr * 0.1},
            ])
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200))
        torch.cuda.empty_cache()
        if DEBUG_MEM and bi % 10 == 0:
            alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"\nstep {bi} alloc={alloc:.0f}MB", flush=True)
        total_loss += loss.item()
        if bi % 200 == 199:
            ppl = torch.exp(torch.tensor(total_loss / (bi + 1))).item()
            lr_now = opt.param_groups[0]['lr']
            route_ent = getattr(model, '_last_route_entropy', 0.0)
            conv_rate = getattr(model, '_conv_rate', 0.0)
            aux_loss = getattr(model, '_last_aux_loss', 0.0)
            cap_rate = getattr(model, '_cap_rate', 0.0)
            gate_ratio = getattr(model, '_gate_ratio', 0.0)
            with torch.no_grad():
                pts = [e.patterns.data for e in model.experts]
                sims = [(pts[i] @ pts[j].T).mean().item() for i in range(4) for j in range(i+1, 4)]
                exp_sim = sum(sims) / len(sims) if sims else 0.0
                if exp_sim > 0.7:
                    model.loser_inhibit = 1.5
                elif exp_sim > 0.5:
                    model.loser_inhibit = 1.0
                elif exp_sim > 0.3:
                    model.loser_inhibit = 0.8
                else:
                    model.loser_inhibit = 0.5
                combined = torch.zeros(1, DM, device=device)
                gate_sum = sum(torch.sigmoid(e.gate_a(combined)).mean().item() for e in model.experts)
                gate_avg = gate_sum / 4
            pbar.set_postfix(ppl=f"{ppl:.1f}", lr=f"{lr_now:.2e}")
            if not os.path.exists(LOG_PATH):
                with open(LOG_PATH, "w", newline="") as f:
                    f.write("epoch,step,ppl,loss,lr,route_ent,exp_sim,gate_avg,grad_norm,conv_rate,aux_loss,cap_rate,gate_ratio\r\n")
            with open(LOG_PATH, "a", newline="") as f:
                f.write(f"{ep},{global_step},{ppl:.1f},{loss.item():.2f},{lr_now:.2e},{route_ent:.4f},{exp_sim:.4f},{gate_avg:.4f},{grad_norm:.4f},{conv_rate:.4f},{aux_loss:.6f},{cap_rate:.4f},{gate_ratio:.4f}\r\n")
        if global_step % 2000 == 0:
            ckpt_mid = {"model": model.state_dict(), "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(), "ep": ep, "ppl": ppl}
            torch.save(ckpt_mid, resume_path)
    ppl = torch.exp(torch.tensor(total_loss / its)).item()
    print(f"  ep {ep}: ppl={ppl:.1f}")
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "ep": ep + 1, "ppl": ppl}
    torch.save(ckpt, resume_path)
    torch.save(ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
