"""Chat with MoHE-RWKV + LoRA. WKV state persistence, token by token."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER
from rina.sample import sample

VOCAB, DM, NP = 65536, 768, 1536
SEQ = 512
LORA_PATH = 'checkpoints/lora_sft/lora_final.pt'
device = 'cuda'

class LoRALayer(nn.Module):
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        out_dim, in_dim = base.weight.shape
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim))
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B)

TARGET_PREFIXES = ['consolidate', 'tmix_r.', 'tmix_k.', 'tmix_v.', 'tmix_a.', 'experts.', 'router']

def attach_lora(module, path=''):
    for name, child in module.named_children():
        full = f"{path}.{name}" if path else name
        if isinstance(child, nn.Linear) and any(full.startswith(p) for p in TARGET_PREFIXES):
            setattr(module, name, LoRALayer(child, 32).to(device))
        else:
            attach_lora(child, full)

CACHE_MODEL = 'checkpoints/chat_model.pt'
if os.path.exists(CACHE_MODEL):
    print("Loading cached model...")
    tokenizer = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
    model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
    state = torch.load(CACHE_MODEL, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=False)
else:
    print("Loading base + LoRA, caching...")
    tokenizer = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
    model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
    state = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
    sd = state.get('model', state.get('model_state_dict', state))
    for k in list(sd.keys()):
        if k.startswith('prev_route') or '_batch_' in k: del sd[k]
    model.load_state_dict(sd, strict=False)
    attach_lora(model)
    lora_sd = torch.load(LORA_PATH, map_location='cpu')
    msd = model.state_dict()
    for k, v in lora_sd.items():
        if k in msd and v.shape == msd[k].shape:
            msd[k].copy_(v.to(device))
    torch.save(model.state_dict(), CACHE_MODEL)
    print("Cached.")
model.eval()
torch.cuda.empty_cache()
print("Ready. Type 'quit' to exit.\n")

def feed(tokens, state):
    """Feed tokens one by one through WKV, return (logits_of_last, new_state)."""
    out = None
    for tid in tokens:
        inp = torch.tensor([[tid]], device=device)
        logits, state = model(inp, wkv_state=state)
        out = logits
    return out, state

state = None
history = ""
while True:
    user = input(">>> ")
    if user.strip().lower() in ("quit", "exit"): break
    prompt = f"User: {user}\nAssistant:"
    # feed prompt tokens (excluding last - we already have it for prediction)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > 0:
        logits, state = feed(prompt_ids[:-1], state)
        inp_last = torch.tensor([[prompt_ids[-1]]], device=device)
        logits, state = model(inp_last, wkv_state=state)

    gen = []
    for _ in range(200):
        tid = sample(logits[0, 0, :]).item()
        gen.append(tid)
        if tid == 0:
            break
        inp = torch.tensor([[tid]], device=device)
        logits, state = model(inp, wkv_state=state)

    reply = tokenizer.decodeBytes(gen).decode('utf-8', errors='replace')
    print(reply.strip())
    # feed reply tokens into state for next turn
    reply_ids = tokenizer.encode(reply.strip())
    _, state = feed(reply_ids, state)
