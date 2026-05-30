"""
Exp 1 (tiny): Slot trust gate — Mini RINA from scratch, dm=32.
Goal: see if slot_read_gate can learn to trust slot in 30 min.
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
slot_table = nn.Embedding(VOCAB, DM).to(device)
slot_read_gate = nn.Linear(DM * 2, 1).to(device)

params = list(cell.parameters()) + list(embed.parameters()) + list(head.parameters()) + \
         list(state_norm.parameters()) + list(slot_table.parameters()) + list(slot_read_gate.parameters())
n = sum(p.numel() for p in params)
print(f"Mini RINA (dm={DM}): {n/1e6:.2f}M params")

opt = torch.optim.AdamW(params, lr=LR)

correct, total = 0, 0

for step in range(STEPS):
    # 80% LM batch: random token sequences
    if random.random() < 0.8:
        x = torch.randint(2, VOCAB, (BS, SEQ), device=device)
        targets = x[:, 1:]
    # 20% NIAH batch: key→value→filler→query→answer
    else:
        k = random.randint(2, VOCAB - 1)
        v = random.randint(2, VOCAB - 1)
        while v == k: v = random.randint(2, VOCAB - 1)
        x = torch.randint(2, VOCAB, (BS, SEQ), device=device)
        kv_pos = random.randint(1, SEQ - 3)
        x[:, 0] = k
        x[:, kv_pos] = k
        x[:, kv_pos + 1] = v
        x[:, -1] = v  # target
        # Write to slot table (differentiable — slot_table is nn.Embedding!)
        with torch.no_grad():
            slot_table.weight.data[k] = slot_table.weight.data[v]
        targets = x[:, 1:]

    # Forward
    emb = embed(x)
    h = torch.zeros(BS, DM, device=device)
    logits = []
    for t in range(SEQ):
        # Slot read with learned gate
        slot_val = slot_table(x[:, t])  # differentiable!
        gate_in = torch.cat([h, emb[:, t, :]], dim=-1)
        gate_val = torch.sigmoid(slot_read_gate(gate_in))
        h = h + gate_val * slot_val

        h = cell(h, emb[:, t, :], step=t)
        logits.append(head(state_norm(h)))
    logits = torch.stack(logits, dim=1)

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), targets.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    # Track slot accuracy
    if step % 100 == 99:
        # Test: generate NIAH and check if model uses slot correctly
        with torch.no_grad():
            k_test = random.randint(2, VOCAB - 1)
            v_test = random.randint(2, VOCAB - 1)
            while v_test == k_test: v_test = random.randint(2, VOCAB - 1)
            x_test = torch.randint(2, VOCAB, (16, SEQ), device=device)
            x_test[:, 0] = k_test
            x_test[:, -2] = k_test  # query
            x_test[:, -1] = v_test  # target
            slot_table.weight.data[k_test] = slot_table.weight.data[v_test]

            emb_t = embed(x_test)
            h_t = torch.zeros(16, DM, device=device)
            for t in range(SEQ):
                slot_val_t = slot_table(x_test[:, t])
                gate_in_t = torch.cat([h_t, emb_t[:, t, :]], dim=-1)
                gate_val_t = torch.sigmoid(slot_read_gate(gate_in_t))
                h_t = h_t + gate_val_t * slot_val_t
                h_t = cell(h_t, emb_t[:, t, :], step=t)
            logits_t = head(state_norm(h_t))
            acc = (logits_t[:, -2].argmax(-1) == v_test).float().mean().item()
            correct += int(acc * 16)
            total += 16
        ppl = torch.exp(loss).item()
        slot_acc = 100 * correct / max(total, 1)
        print(f"  step {step+1}: ppl={ppl:.2f} slot_acc={slot_acc:.0f}%")

print(f"\nFinal slot_acc: {100 * correct / max(total, 1):.1f}% (random: ~0.02%)")
