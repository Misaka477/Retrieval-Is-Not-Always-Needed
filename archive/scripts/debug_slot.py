"""Debug: does slot injection actually change model output?"""
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rina_v3 import RINAModel

device = 'cuda'
model = RINAModel(vocab_size=23, d_model=64, n_patterns=1024, beta=0.5, n_iter=3).to(device)
model.eval()

model.slot.insert(5, 0, 17)

seq = [0] * 10 + [5]
x = torch.tensor([seq], device=device)

with torch.no_grad():
    logits_with = model(x)
    prob_with = F.softmax(logits_with[0, -1], dim=-1)

model.slot.clear()

with torch.no_grad():
    logits_without = model(x)
    prob_without = F.softmax(logits_without[0, -1], dim=-1)

print(f"With slot:  top-5 ids={prob_with.topk(5).indices.tolist()}, val={prob_with[17].item():.4f}")
print(f"Without slot: top-5 ids={prob_without.topk(5).indices.tolist()}, val={prob_without[17].item():.4f}")
diff = (prob_with - prob_without).abs().sum().item()
print(f"Diff: {diff:.4f} {'✅ SIGNIFICANT' if diff > 0.1 else '❌ negligible'}")
