"""Evaluate diffusion head with more steps and better thresholds."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, math
from rina.model import MoHERWKV

device = 'cuda'
VOCAB, DM = 65536, 768

print('Loading backbone + trained head...')
model = MoHERWKV(VOCAB, DM, 1536, n_experts=12, topk=2).to(device)
ckpt = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
s = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(s, strict=False)
for p in model.parameters():
    p.requires_grad_(False)
model.head = nn.Linear(DM, VOCAB, bias=True).to(device)
model.head.load_state_dict(torch.load('checkpoints/diff_head.pt', map_location='cpu', weights_only=False)['head'])
model.eval()
print('Loaded diff_head.pt')

def generate_diffusion(model, n_tokens=32, steps=30, temp_start=1.0, temp_end=0.01):
    with torch.no_grad():
        x = torch.randint(10, VOCAB, (1, n_tokens), device=device)
        for step in range(steps):
            temp = temp_start + (temp_end - temp_start) * step / steps
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            p = torch.softmax(logits / max(temp, 0.001), dim=-1)
            max_p, _ = p.max(dim=-1)
            thresh = 0.5 + 0.4 * math.sin(math.pi * step / steps)
            keep = max_p > thresh
            resample = torch.multinomial(p.view(-1, VOCAB), 1).view(1, n_tokens)
            x = torch.where(keep, x, resample)
        return x[0].tolist(), p

for label, hd, steps in [
    ('Trained head, 30 steps', 'diff_head.pt', 30),
    ('Trained head, 100 steps', 'diff_head.pt', 100),
    ('Original head, 30 steps', None, 30),
    ('Original head, 100 steps', None, 100),
]:
    if hd:
        model.head = nn.Linear(DM, VOCAB, bias=True).to(device)
        model.head.load_state_dict(torch.load(f'checkpoints/{hd}', map_location='cpu', weights_only=False)['head'])
    else:
        model.head = nn.Linear(DM, VOCAB, bias=True).to(device)
        model.head.weight.data.copy_(model.embed.weight.data)
        model.head.bias.data.fill_(-10.8)

    out, probs = generate_diffusion(model, steps=steps)
    max_ps = probs.max(dim=-1).values
    avg_conf = max_ps.mean().item()
    uni = len(set(out))
    print(f'\n[{label}]')
    print(f'  Unique: {uni}/{len(out)}')
    print(f'  Avg confidence: {avg_conf:.4f}')
    print(f'  Output: {out}')
