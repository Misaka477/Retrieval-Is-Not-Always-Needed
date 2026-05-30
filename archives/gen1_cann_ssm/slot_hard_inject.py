"""
Exp 1b: Slot hard injection — no gate, always inject slot.
Mini RINA (dm=32), mixed LM + NIAH data.
If model learns to use slot (ppl drops), slot mechanism works.
If not, something fundamental is wrong.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F, random

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 32, 128
SEQ, BS = 32, 4
LR = 1e-3; STEPS = 2000

from rina.cell import TemporalSNNCell
cell = TemporalSNNCell(DM, NP, error_threshold=0.5, hebbian_lr=0.0).to(device)
embed = nn.Embedding(VOCAB, DM).to(device)
head = nn.Linear(DM, VOCAB).to(device)
state_norm = nn.LayerNorm(DM).to(device)
# Differentiable slot table — initialized to near-zero
slot_table = nn.Embedding(VOCAB, DM).to(device)
nn.init.normal_(slot_table.weight, mean=0.0, std=0.001)

params = list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()) + \
         list(state_norm.parameters()) + list(slot_table.parameters())
n = sum(p.numel() for p in params)
print(f"Mini RINA (dm={DM}): {n/1e6:.2f}M params")
opt = torch.optim.AdamW(params, lr=LR)

correct, total = 0, 0
query_pos = -3  # position where query key is placed

for step in range(STEPS):
    if random.random() < 0.8:
        x = torch.randint(2, VOCAB, (BS, SEQ), device=device)
        targets = x[:, 1:]
        is_niah = False
    else:
        k = random.randint(2, VOCAB - 1)
        v = random.randint(2, VOCAB - 1)
        while v == k: v = random.randint(2, VOCAB - 1)
        x = torch.randint(2, VOCAB, (BS, SEQ), device=device)
        kv_pos = random.randint(1, SEQ - 4)
        x[:, kv_pos] = k
        x[:, kv_pos + 1] = v
        x[:, query_pos] = k  # query
        x[:, query_pos + 1] = v  # target
        with torch.no_grad():
            slot_table.weight.data[k] = embed(torch.tensor([v], device=device)).squeeze(0) * 0.1
        targets = x[:, 1:]
        is_niah = True

    emb = embed(x)
    h = torch.zeros(BS, DM, device=device)
    logits = []
    for t in range(SEQ):
        # Only inject slot at query position during NIAH batches
        if is_niah and t == query_pos:
            h = h + slot_table(x[:, t])  # inject stored value
        h = cell(h, emb[:, t, :], step=t)
        logits.append(head(state_norm(h)))
    logits = torch.stack(logits, dim=1)

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), targets.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    if step % 200 == 199:
        with torch.no_grad():
            k_test = random.randint(2, VOCAB - 1)
            v_test = random.randint(2, VOCAB - 1)
            while v_test == k_test: v_test = random.randint(2, VOCAB - 1)
            x_test = torch.randint(2, VOCAB, (16, SEQ), device=device)
            x_test[:, query_pos] = k_test
            x_test[:, query_pos + 1] = v_test
            slot_table.weight.data[k_test] = embed(torch.tensor([v_test], device=device)).squeeze(0) * 0.1
            emb_t = embed(x_test)
            h_t = torch.zeros(16, DM, device=device)
            for t in range(SEQ):
                if t == query_pos:
                    h_t = h_t + slot_table(x_test[:, t])
                h_t = cell(h_t, emb_t[:, t, :], step=t)
            logits_t = head(state_norm(h_t))
            acc = (logits_t[:, query_pos].argmax(-1) == v_test).float().mean().item()
            correct += int(acc * 16)
            total += 16
        slot_acc = 100 * correct / max(total, 1)
        print(f"  step {step+1}: ppl={torch.exp(loss).item():.1f} slot_acc={slot_acc:.0f}%")

print(f"\nFinal: slot_acc={100 * correct / max(total, 1):.1f}%")
