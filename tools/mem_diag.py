"""Memory leak diagnostic: run the EXACT same training loop and track per-step allocations."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rina.mohe import MoHE  # type: ignore

device = "cuda"
VOCAB, DM, SEQ, BS, NE = 50257, 256, 64, 8, 4
print(f"Config: VOCAB={VOCAB}, DM={DM}, SEQ={SEQ}, BS={BS}, NE={NE}")

model = MoHE(VOCAB, DM, 512, n_experts=NE).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randint(0, VOCAB, (BS, SEQ), device=device)

for step in range(50):
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    loss.backward()
    model.finish_training_step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    gc.collect()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    if step <= 5 or step % 10 == 0:
        print(f"Step {step:3d}: alloc={alloc:.0f}MB  reserved={reserved:.0f}MB")
    if reserved > 7000:
        print(f"ERROR: reserved > 7GB at step {step}, OOM imminent")
        break

print(f"Final: alloc={alloc:.0f}MB  reserved={reserved:.0f}MB")
