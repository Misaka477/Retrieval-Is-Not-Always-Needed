"""Quick benchmark: GSM8K math + code generation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset
import torch, torch.nn as nn
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER


device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
MAX_GEN = 100

# ── Load model + LoRA ──
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

print("Loading...")
tokenizer = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
base = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
sd = base.get('model', base.get('model_state_dict', base))
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k: del sd[k]
model.load_state_dict(sd, strict=False)
attach_lora(model)
lora_sd = torch.load('checkpoints/lora_sft/lora_final.pt', map_location='cpu')
msd = model.state_dict()
for k, v in lora_sd.items():
    if k in msd and v.shape == msd[k].shape:
        msd[k].copy_(v.to(device))
model.eval()

def generate(prompt):
    tokens = tokenizer.encode(prompt)
    gen = tokens[:]
    with torch.no_grad():
        for _ in range(MAX_GEN):
            out = model(torch.tensor([gen[-512:]], device=device))
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :]
            tid = torch.argmax(logits).item()
            gen.append(tid)
            if tid == 0: break
    return tokenizer.decodeBytes(gen).decode('utf-8', errors='replace')

# ── GSM8K ──
print("Loading GSM8K test set...")
gsm = load_dataset("gsm8k", "main", split="test", streaming=True)
correct = 0; total = 0
for i, item in enumerate(gsm):
    if i >= 20: break
    question = item["question"]
    answer = item["answer"].split("####")[-1].strip()
    prompt = f"<|user|>\nSolve: {question}\n<|assistant|>\n"
    resp = generate(prompt)
    has_answer = answer in resp.replace(",", "")
    if has_answer: correct += 1
    if has_answer or i < 3:
        print(f"\n      Resp: {resp[:200]}")
    total += 1
    print(f"  [{correct}/{total}] {'✓' if has_answer else '✗'} {question[:60]}...")
print(f"\nGSM8K ({total} samples): {correct}/{total} = {correct/total*100:.0f}%")

# ── Simple code test ──
print("\n--- Code ---")
tests = [
    ("Write a function to check if a number is even",
     "def is_even"),
    ("Write a function to add two numbers",
     "def add"),
]
for desc, expected in tests:
    prompt = f"<|user|>\n{desc}\n<|assistant|>\n"
    resp = generate(prompt)
    has_def = expected in resp
    print(f"  {'✓' if has_def else '✗'} {desc}")
    if has_def:
        print(f"    {resp[:100]}...")
