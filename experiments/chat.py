"""Interactive chat with MoHE-RWKV, WKV state persistence between turns."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER
from rina.sample import sample

VOCAB, DM, NP = 65536, 768, 1536
SEQ = 512
device = 'cuda'

print("Loading...")
tokenizer = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
state = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
sd = state.get('model', state.get('model_state_dict', state))
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k: del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
print("Ready (WKV state persistence enabled). Type 'quit' to exit.\n")

wkv_state = None
while True:
    user = input(">>> ")
    if user.strip().lower() in ("quit", "exit"): break
    prompt = f"User: {user}\nAssistant:"
    tokens = tokenizer.encode(prompt)
    gen = tokens[:]
    with torch.no_grad():
        for _ in range(200):
            inp = torch.tensor([gen[-SEQ:]], device=device)
            logits, wkv_state = model(inp, wkv_state=wkv_state)
            tid = sample(logits[0, -1, :]).item()
            gen.append(tid)
            if tid == 0:
                break
    reply = tokenizer.decodeBytes(gen[len(tokens):]).decode('utf-8', errors='replace')
    print(reply.strip())
    # prepend assistant reply as context for next turn
    reply_tokens = tokenizer.encode(reply.strip())
    # encode current turn as continuation and process through WKV
    cont = tokenizer.encode(f"User: {user}\nAssistant: {reply.strip()}\n")
    with torch.no_grad():
        for tid in cont:
            inp = torch.tensor([[tid]], device=device)
            _, wkv_state = model(inp, wkv_state=wkv_state)
