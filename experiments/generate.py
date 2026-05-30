"""Generate text from MoHE-RWKV checkpoint using RWKV tokenizer."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER
from rina.sample import sample

device = 'cuda'
CKPT = 'checkpoints/mohe_transferred_latest.pt'
VOCAB_FILE = 'checkpoints/rwkv_vocab_v20230424.txt'
SEQ = 512
VOCAB, DM, NP = 65536, 768, 1536

print("Loading tokenizer...")
tokenizer = TRIE_TOKENIZER(VOCAB_FILE)

print("Loading model...")
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
state = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = state.get('model', state.get('model_state_dict', state))
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k:
        del sd[k]
# align shape mismatches caused by architecture changes
msd = model.state_dict()
for k in list(sd.keys()):
    if k in msd and sd[k].shape != msd[k].shape:
        del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
print(f"Loaded step {state.get('step', '?')}")

prompts = [
    "The meaning of life is",
    "def fibonacci(n):",
    "Once upon a time,",
]

with torch.no_grad():
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        gen = tokens[:]
        for _ in range(200):
            inp = torch.tensor([gen[-SEQ:]], device=device)
            logits = model(inp)[0, -1, :]
            gen.append(sample(logits).item())
        text = tokenizer.decodeBytes(gen).decode('utf-8', errors='replace')
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text[:500]}")
        import sys; sys.stdout.flush()
