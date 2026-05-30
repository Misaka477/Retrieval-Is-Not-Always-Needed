"""
3.7M RINA full training: mixed FineWeb+StarCoder, seq=256, differentiable slot.
All components trainable. CSV logging, resume, 2000-step checkpoints.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import io, signal, gc
def cleanup(sig, frame):
    print("\nSIGINT — saving checkpoint...", flush=True)
    if 'model' in dir(): torch.save(model.state_dict(), CKPT_DIR + "/rina_37m_interrupted.pt")
    torch.cuda.synchronize(); sys.exit(0)
signal.signal(signal.SIGINT, cleanup)

from tokenizers import Tokenizer
from datasets import load_dataset, concatenate_datasets
import torch, torch.nn as nn, torch.nn.functional as F, random
from rina import TemporalSNNModel
from rina.drift import DriftTracker
from tqdm import tqdm
import numpy as np

device = "cuda"; torch.manual_seed(42); random.seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

# Config
VOCAB, DM, NP = 50257, 256, 1024
SEQ, BS = 256, 8
LR = 3e-4; EPOCHS = 5; SUBSAMPLE = 4
MAX_TOKENS = 200_000_000
CKPT_NAME = "rina_37m"
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
RESUME_PATH = os.path.join(CKPT_DIR, f"{CKPT_NAME}_resume.pt")

print(f"Config: dm={DM} np={NP} seq={SEQ} bs={BS} lr={LR} ep={EPOCHS} max_tokens={MAX_TOKENS:,}")

# Build model
print("Building 3.7M RINA with differentiable slot...")
model = TemporalSNNModel(VOCAB, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=2, error_threshold=0.5,
                          hebbian_lr=LR, hebbian_decay=1.0, inhibition_threshold=0.0,
                          n_slots=VOCAB).to(device)

# Differentiable slot Embedding (separate from model's buffer)
slot_embed = nn.Embedding(VOCAB, DM).to(device)
nn.init.normal_(slot_embed.weight, mean=0.0, std=0.01)
slot_read_gate = nn.Linear(DM * 2, 1).to(device)
write_net = nn.Linear(DM * 2, 1).to(device)
slot_proj = nn.Linear(DM, DM).to(device)

model.slot_table.zero_()  # freeze model's buffer

n = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in slot_embed.parameters()) + sum(p.numel() for p in slot_read_gate.parameters()) + sum(p.numel() for p in write_net.parameters()) + sum(p.numel() for p in slot_proj.parameters())
print(f"  params: {n/1e6:.3f}M ({sum(p.numel() for p in model.parameters())/1e6:.3f}M + slot {sum(p.numel() for p in slot_embed.parameters())/1e3:.0f}K + gate {sum(p.numel() for p in slot_read_gate.parameters())/1e3:.0f}K + write {sum(p.numel() for p in write_net.parameters())/1e3:.0f}K + proj {sum(p.numel() for p in slot_proj.parameters())/1e3:.0f}K)")

drift = DriftTracker(compute_coverage=True)
opt = torch.optim.AdamW(list(model.parameters()) + list(slot_embed.parameters()) + list(slot_read_gate.parameters()) + list(write_net.parameters()) + list(slot_proj.parameters()), lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler()

# ── GPT-2 Tokenizer ──
print("Loading GPT-2 tokenizer...")
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("gpt2")
print(f"  vocab: {tok.get_vocab_size()}")

# Data
CACHE_FW = os.path.join(CKPT_DIR, "fweb_tokens.npy")
CACHE_SC = os.path.join(CKPT_DIR, "sc_tokens.npy")

def np_save_chunked(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    data = buf.getvalue()
    CHUNK = 64 * 1024 * 1024
    with open(path, 'wb') as f:
        for i in range(0, len(data), CHUNK):
            f.write(data[i:i + CHUNK])

# Clean up .pt cache remnants from previous failed torch.save
for p in [CACHE_FW.replace(".npy", ".pt"), CACHE_SC.replace(".npy", ".pt")]:
    if os.path.exists(p): os.remove(p)

# Clean up any corrupt partial files from previous runs
for p in [CACHE_FW, CACHE_SC, CACHE_FW + ".part", CACHE_SC + ".part"]:
    if os.path.exists(p):
        try:
            np.load(p)
        except Exception:
            os.remove(p)

# Migrate legacy .pt.npy → .npy  (cleanly, no big-write)
for legacy, new in [(CACHE_FW[:-4] + ".pt.npy", CACHE_FW),
                    (CACHE_SC[:-4] + ".pt.npy", CACHE_SC)]:
    if os.path.exists(legacy) and not os.path.exists(new):
        try:
            data = np.load(legacy)
            np_save_chunked(new, data)
            os.remove(legacy)
            print(f"  migrated {len(data):,} tokens: {os.path.basename(legacy)} → {os.path.basename(new)}")
        except Exception:
            os.remove(legacy)

def load_or_tokenize(src_name, src_config, split, cache_path, desc, max_tokens=MAX_TOKENS):
    part_path = cache_path + ".part"
    if os.path.exists(cache_path):
        ids = torch.from_numpy(np.load(cache_path))
        print(f"  {desc}: {len(ids):,} tokens (cached)")
        return ids
    
    pos = 0
    arr = np.empty(max_tokens, dtype=np.int32)
    if os.path.exists(part_path):
        partial = np.load(part_path)
        pos = len(partial); arr[:pos] = partial
        print(f"  {desc}: resuming from {pos:,} tokens")
        os.remove(part_path)
    
    print(f"  Loading {desc}...")
    ds = load_dataset(src_name, src_config, split=split, streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=42)
    batch = []; save_interval = 50_000_000
    for sample in tqdm(ds, desc=f"  Tokenizing {desc}"):
        text = sample.get("text") or sample.get("content", "")
        if len(text) < 200: continue
        batch.append(text)
        if len(batch) >= 500:
            for enc in tok.encode_batch(batch):
                tids = enc.ids[:SEQ * 500]
                if len(tids) >= SEQ:
                    take = min(len(tids), max_tokens - pos)
                    arr[pos:pos+take] = tids[:take]; pos += take
                    if pos >= max_tokens: break
            if pos // save_interval > (pos - len(batch)) // save_interval:
                np_save_chunked(part_path, arr[:pos])
            batch = []
            if pos >= max_tokens: break
    if batch:
        for enc in tok.encode_batch(batch):
            tids = enc.ids[:SEQ * 500]
            if len(tids) >= SEQ:
                take = min(len(tids), max_tokens - pos)
                arr[pos:pos+take] = tids[:take]; pos += take
                if pos >= max_tokens: break
    ids = torch.tensor(arr[:pos], dtype=torch.long)
    np_save_chunked(cache_path, ids.numpy())
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"  {desc}: {len(ids):,} tokens (cached)")
    return ids

ids_fw = load_or_tokenize("HuggingFaceFW/fineweb", "sample-10BT", "train", CACHE_FW, "FineWeb")
ids_sc = load_or_tokenize("bigcode/starcoderdata", None, "train", CACHE_SC, "StarCoder")

# Mix: 80% FW + 20% SC
mix_ratio = 0.8
max_len = min(len(ids_fw), len(ids_sc) * 4)
ids_mix = torch.zeros(max_len, dtype=torch.long)
n_fw = int(max_len * mix_ratio)
ids_mix[:n_fw] = ids_fw[:n_fw]
ids_mix[n_fw:] = ids_sc[:(max_len - n_fw)]
ids = ids_mix

num_batches = (len(ids) - 1) // (BS * SEQ)
train_its = num_batches // SUBSAMPLE
print(f"  total tokens: {len(ids):,}, batches/epoch: {train_its}")

# Resume
start_ep = 1; resume_steps = 0
if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    slot_embed.load_state_dict(ckpt["slot_embed"])
    slot_read_gate.load_state_dict(ckpt["slot_read_gate"])
    if "write_net" in ckpt:
        write_net.load_state_dict(ckpt["write_net"])
        slot_proj.load_state_dict(ckpt["slot_proj"])
    opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
    start_ep = ckpt["ep"]; resume_steps = ckpt.get("train_steps", 0)
    print(f"  resume ep {start_ep}, step {resume_steps}")

torch.cuda.empty_cache()

# ── 16K BPE Tokenizer ──
torch.cuda.empty_cache()

print(f"Context: dm={DM} np={NP} seq={SEQ} bs={BS} lr={LR} ep={EPOCHS} max_tokens={MAX_TOKENS:,}")

def make_niah_batch(bs, seq, vocab_size, text_pool):
    """Insert key→value pair into real text at random position. Query at last position."""
    x = torch.randint(vocab_size, (bs, seq))
    keys, vals = [], []
    for b in range(bs):
        # Use a random text segment as background
        start = random.randint(0, len(text_pool) - seq - 1)
        for t in range(seq):
            x[b, t] = text_pool[start + t]
        k = random.randint(2, vocab_size - 1)
        v = random.randint(2, vocab_size - 1)
        while v == k: v = random.randint(2, vocab_size - 1)
        kv_pos = random.randint(1, seq - 4)
        x[b, kv_pos] = k
        x[b, kv_pos + 1] = v
        x[b, -2] = k  # query
        x[b, -1] = v  # target
        keys.append(k); vals.append(v)
    return x, keys, vals

slot_correct = 0; slot_total = 0
text_pool = ids.cpu().numpy() if isinstance(ids, torch.Tensor) else ids

for ep in range(start_ep, EPOCHS + 1):
    model.train(); total_loss = 0.0
    perm = torch.randperm(len(ids) - BS * SEQ)
    pbar = tqdm(range(train_its), desc=f"ep {ep}/{EPOCHS}")
    train_steps = resume_steps if ep == start_ep else 0
    skip_until = resume_steps if ep == start_ep else 0
    drift_anchor = model.cell.patterns.detach().clone()

    log_file = os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.csv")
    expected_header = "step,loss,ppl,att_pct,dead_pct,norms_mu,cos_drift,frob_drift,r95,slot_acc,gate_avg,grad_norm,lr"
    needs_header = True
    if os.path.exists(log_file):
        with open(log_file, "r", newline="") as f:
            first = f.readline().rstrip("\r\n")
            if first == expected_header:
                needs_header = False
    if needs_header:
        with open(log_file, "w", newline="") as f:
            f.write(expected_header + "\r\n")

    for bi in pbar:
        train_steps += 1
        if train_steps <= skip_until: continue
        gate_act_sum = torch.zeros(1, device=device); gate_act_cnt = 0

        is_niah = (bi % 5 == 0)
        keys, vals = None, None
        if is_niah:
            x, keys, vals = make_niah_batch(BS, SEQ, VOCAB, text_pool)
            x = x.to(device)
        else:
            start = perm[(bi * SUBSAMPLE) % len(perm)]
            x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)

        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            bsz, seq_len = x.shape
            emb = model.embed(x)
            slot_vals = slot_embed(x)
            h = torch.zeros(bsz, DM, device=device)
            logits = []
            for t in range(seq_len):
                slot_val = slot_vals[:, t, :]
                gate_in = torch.cat([h, emb[:, t, :]], dim=-1)
                gate_val = torch.sigmoid(slot_read_gate(gate_in))
                gate_act_sum += gate_val.mean().detach(); gate_act_cnt += 1
                h = model.cell(h + gate_val * slot_val * 0.05, emb[:, t, :], step=t)
                # Soft write: augment h with content from current state
                write_gate = torch.sigmoid(write_net(gate_in.detach()))
                write_content = slot_proj(h)
                h = h + write_gate * write_content * 0.1
                logits.append(model.head(model.state_norm(h)))
            logits = torch.stack(logits, dim=1)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
            if is_niah:
                key_t = torch.tensor(keys, device=device)
                val_t = torch.tensor(vals, device=device)
                target_emb = slot_proj(model.embed(val_t))
                stored = slot_embed(key_t)
                niah_loss = F.mse_loss(stored, target_emb)
                loss = loss + niah_loss * 0.1

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(slot_embed.parameters()) + list(slot_read_gate.parameters()) + list(write_net.parameters()) + list(slot_proj.parameters()), 1.0).item()
        scaler.step(opt); scaler.update()
        total_loss += loss.item()

        # Track slot accuracy on NIAH batches
        if is_niah:
            with torch.no_grad():
                preds = logits[:, -2].argmax(-1)
                for b in range(BS):
                    slot_total += 1
                    if preds[b].item() == vals[b]:
                        slot_correct += 1

        if train_steps % 200 == 0:
            ppl = torch.exp(torch.tensor(total_loss / (train_steps - skip_until))).item()
            att = model.get_att_rate()
            P = model.cell.patterns.detach(); norms = P.norm(dim=-1)
            dead = (norms < 0.01).float().mean().item(); mu = norms.mean().item()
            Pn = P / norms.unsqueeze(-1).clamp(min=1e-8)
            An = drift_anchor / drift_anchor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cos = (Pn * An).sum(dim=-1).mean().item()
            frob = (P - drift_anchor).norm().item() / drift_anchor.norm().item()
            _, S, _ = torch.linalg.svd(P.float(), full_matrices=False)
            cv = (S.cumsum(0) / S.sum()).tolist()
            r95 = next((i for i, v in enumerate(cv) if v >= 0.95), len(S))
            sa = 100 * slot_correct / max(slot_total, 1)
            ga = (gate_act_sum / max(gate_act_cnt, 1)).item()
            pbar.set_postfix(ppl=f"{ppl:.1f}", att=f"{att*100:.0f}%", slot=f"{sa:.0f}%", gate=f"{ga:.2f}")
            with open(log_file, "a", newline="") as f:
                lr_now = opt.param_groups[0]["lr"]
                f.write(f"{train_steps},{loss.item():.2f},{ppl:.2f},{att*100:.2f},{dead*100:.2f},{mu:.2f},{cos:.4f},{frob:.4f},{r95},{sa:.2f},{ga:.4f},{grad_norm:.4f},{lr_now:.2e}\r\n")
                f.flush()

            if train_steps % 2000 == 0:
                ckpt = {"model": model.state_dict(), "slot_embed": slot_embed.state_dict(),
                        "slot_read_gate": slot_read_gate.state_dict(),
                        "write_net": write_net.state_dict(), "slot_proj": slot_proj.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "ep": ep,
                        "ppl": ppl, "train_steps": train_steps}
                torch.save(ckpt, RESUME_PATH)
                torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}_st{train_steps}.pt"))

    effective = train_steps - skip_until
    ppl = torch.exp(torch.tensor(total_loss / effective)).item()
    d = drift.step(model.cell.patterns)
    print(f"ep {ep}: ppl={ppl:.1f} cos={d['avg_cos']:.4f} frob={d['frob_drift']:.4f} "
          f"mu={d['norms_mean']:.2f} dead={d['dead_frac']*100:.0f}% r95={d['eff_rank_95']}")

    ckpt = {"model": model.state_dict(), "slot_embed": slot_embed.state_dict(),
            "slot_read_gate": slot_read_gate.state_dict(),
            "write_net": write_net.state_dict(), "slot_proj": slot_proj.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(), "ep": ep + 1,
            "ppl": ppl, "train_steps": 0}
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{CKPT_NAME}_ep{ep}.pt"))
    torch.save(ckpt, RESUME_PATH)

print(f"\nDone. Final ppl={ppl:.1f}")
