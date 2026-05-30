import sys, os; sys.path.insert(0, '.')
import torch
from rina.mohe import MoHE
from tokenizers import Tokenizer
from rina.sample import sample

device = 'cuda'
VOCAB, DM, NP = 50257, 1024, 2048
ckpt = torch.load('checkpoints/mohe_91m_resume.pt', map_location=device, weights_only=False)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k:
        del sd[k]
model = MoHE(VOCAB, DM, NP, n_experts=4, aux_loss_weight=0.5, route_noise=0.0, topk=2).to(device)
model.load_state_dict(sd, strict=False)
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f'Params: {n/1e6:.2f}M')
print(f'Checkpoint ppl: {ckpt.get("ppl", "?")}')

tok = Tokenizer.from_pretrained('gpt2')
prompts = ['The meaning of life is', 'def fibonacci(n):', 'Once upon a time']
for p in prompts:
    ids = tok.encode(p).ids[:8]
    gen = ids[:]
    with torch.no_grad():
        for _ in range(60):
            inp = torch.tensor([gen[-128:]], device=device)
            logits = model(inp)[0, -1, :]
            gen.append(sample(logits, temp=0.7, top_p=0.9).item())
    text = tok.decode(gen)
    print(f'\n--- {p} ---\n{text[:400]}')
