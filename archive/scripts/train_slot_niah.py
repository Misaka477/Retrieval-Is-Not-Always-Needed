"""
阶段 B：纯 NIAH fine-tune — 从混合训练 checkpoint 加载，专攻 slot 准确率。
预期 slot_acc 从 ~9% 推至 60-80%。

用法：
  python scripts/train_slot_niah.py                    # 从 slot 混合训练最终 checkpoint
  python scripts/train_slot_niah.py --resume            # 从本脚本的 resume checkpoint 续
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch, torch.nn.functional as F, random, glob
from rina import TemporalSNNModel
from rina.niah import NIAHGenerator
from tqdm import tqdm

device = "cuda"; torch.manual_seed(42)
V, DM, NP, AE = 4096, 840, 4096, 2
SEQ, BS = 64, 8
LR = 3e-4; EPOCHS = 2

# Auto-detect latest slot checkpoint
candidates = sorted(glob.glob("checkpoints/cann_snn15m_v2_slot_ep*.pt"))
CKPT_SOURCE = candidates[-1] if candidates else "checkpoints/cann_snn15m_v2_slot_ep8.pt"
print(f"Loading checkpoint: {CKPT_SOURCE}", flush=True)
sd = torch.load(CKPT_SOURCE, map_location=device, weights_only=False)
model = TemporalSNNModel(V, d_model=DM, n_patterns=NP, beta=0.5,
                          attract_every=AE, error_threshold=1.0,
                          hebbian_lr=0.0, inhibition_threshold=0.0,
                          n_slots=V).to(device)
missing, unexpected = model.load_state_dict(sd["model"], strict=False)
print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}", flush=True)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,} ({n/1e6:.1f}M)", flush=True)

print("Loading WikiText-103 paragraphs for NIAH background...", flush=True)
gen = NIAHGenerator()
texts = gen.load_wikitext_paragraphs(num_segments=50000)
print(f"  paragraphs: {len(texts)}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler()
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

start_ep = 1
if "--resume" in sys.argv:
    resume_path = os.path.join(CKPT_DIR, "cann_slotniah_resume.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["ep"]
        print(f"  resume from ep {start_ep}", flush=True)

batches_per_epoch = 2000  # ~1h per epoch
print(f"batches/epoch: {batches_per_epoch}\n", flush=True)

for ep in range(start_ep, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(range(batches_per_epoch), desc=f"ep {ep}/{EPOCHS}")

    for bi in pbar:
        gen.rng = random.Random(42 + bi + ep * batches_per_epoch)
        xb, keys, vals = gen.make_batch(texts, BS, SEQ)
        xb = xb.to(device)
        model.slot_table.zero_()
        for i in range(BS):
            model.slot_write(keys[i], vals[i])

        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(xb)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), xb[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        for i in range(BS):
            total += 1
            if logits[i, -1].argmax().item() == vals[i]:
                correct += 1

        if bi % 100 == 0 and bi > 0:
            acc = 100 * correct / total
            avg_loss = total_loss / bi
            pbar.set_postfix(loss=f"{avg_loss:.3f}", slot=f"{acc:.0f}%")

    acc = 100 * correct / total
    avg_loss = total_loss / batches_per_epoch
    print(f"ep {ep:2d}: loss={avg_loss:.3f} slot_acc={acc:.1f}%")

    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "ep": ep,
    }, os.path.join(CKPT_DIR, "cann_slotniah_ep{ep}.pt"))
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "ep": ep,
    }, os.path.join(CKPT_DIR, "cann_slotniah_resume.pt"))

torch.save(model.state_dict(), os.path.join(CKPT_DIR, "cann_slotniah_final.pt"))
print(f"\nDone. slot_acc={acc:.1f}%")
