"""
Exp 4: Slot at long context (seq=512).
Take code-seq256 checkpoint. Write key→value at position 10,
query at position 500. Measure recall.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, random
from rina import TemporalSNNModel

device = "cuda"; torch.manual_seed(42); random.seed(42)
VOCAB, DM, NP = 4096, 840, 4096
SEQ = 512
CKPT = "checkpoints/code_seq256_resume.pt"

print(f"Loading {CKPT}...")
sd = torch.load(CKPT, map_location=device, weights_only=False)
m = TemporalSNNModel(VOCAB, d_model=DM, n_patterns=NP, beta=0.5,
                      attract_every=2, error_threshold=1.0,
                      hebbian_lr=0.0, inhibition_threshold=0.0,
                      n_slots=VOCAB).to(device)
m.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
m.eval()
print(f"  {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")

# Test: write key→value, query 500 tokens later
N_TESTS = 50
correct_exact = 0
correct_attn = 0

from rina import TemporalSNNCell
attn_slot_keys = torch.randn(64, DM, device=device)
attn_slot_vals = torch.randn(64, DM, device=device)

for test in range(N_TESTS):
    x = torch.randint(2, VOCAB, (1, SEQ), device=device)
    key = random.randint(2, VOCAB - 1)
    val = random.randint(2, VOCAB - 1)
    while val == key: val = random.randint(2, VOCAB - 1)
    
    # Insert key at position 10, query at position 500
    x[0, 10] = key
    x[0, 11] = val
    x[0, 500] = key  # query
    
    # Write to slot table (current approach)
    m.slot_write(key, val)
    
    # Forward
    with torch.no_grad():
        logits = m(x)
        
        # Test 1: Exact token match slot
        pred_exact = logits[0, 500].argmax().item()
        if pred_exact == val:
            correct_exact += 1
        
        # Test 2: Attention-based slot (manual, for comparison)
        h = torch.zeros(1, DM, device=device)
        emb = m.embed(x)
        for t in range(SEQ):
            # Current slot injection at every position
            h_in = h + m.slot_table[x[:, t]] if m.n_slots > 0 else h
            h = m.cell(h_in, emb[:, t, :], step=t)
        logit = m.head(m.state_norm(h))

print(f"\nSeq=512 NIAH recall ({N_TESTS} tests):")
print(f"  Current slot_table (exact match):  {correct_exact}/{N_TESTS} ({100*correct_exact/N_TESTS:.0f}%)")
print(f"\nInterpretation:")
if correct_exact > 0:
    print(f"  Slot DOES work at long context! {100*correct_exact/N_TESTS:.0f}% recall.")
else:
    print(f"  Slot does NOT work at long context.")
    print(f"  The current slot_table with .data assignment is not differentiable.")
    print(f"  The model never learned to trust slot during training (slot_acc=0%).")
