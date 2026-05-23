"""Memory leak diagnostic: exact replica of mohe_large_run.py training loop."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from rina.mohe import MoHE

device = "cuda"
VOCAB, DM, SEQ, BS, NE = 50257, 256, 64, 8, 4
print(f"Config: VOCAB={VOCAB} DM={DM} SEQ={SEQ} BS={BS} NE={NE}")

model = MoHE(VOCAB, DM, 512, n_experts=NE).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step/500))

# Simulate loaded token data (200M int32 mmap tensor equivalent)
ids = torch.randint(0, 50257, (200_000_000,), dtype=torch.long)

total_loss = 0.0
for bi in range(200):
    start = (bi * 8) % (len(ids) - BS * SEQ)
    x = ids[start:start + BS * SEQ].view(BS, SEQ).to(device)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    if torch.isnan(loss) or torch.isinf(loss):
        scheduler.step()
        continue
    loss.backward()
    model.finish_training_step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); scheduler.step()
    total_loss += loss.item()

    if bi % 10 == 0:
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Step {bi:3d}: alloc={alloc:.0f}MB  reserved={reserved:.0f}MB  loss={loss.item():.2f}")
    if bi >= 50 and reserved > 2000:
        print(f"⚠ reserved {reserved:.0f}MB > 2GB at step {bi}")

print(f"Done. {bi+1} steps. alloc={alloc:.0f}MB reserved={reserved:.0f}MB")
