"""LoRA SFT on MoHE-RWKV 109M. Saves adapter weights only."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rina import MoHERWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
LR = 1e-4; SEQ = 1024; BSZ = 2; N_STEPS = 2000; SAVE_EVERY = 200; LORA_R = 32

CKPT = 'checkpoints/mohe_transferred_latest.pt'
SFT_DATA = 'experiments/checkpoints/sft_data/sft_filtered.jsonl'
OUT_DIR = 'checkpoints/lora_sft'; os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')

print("Loading model (base frozen)...")
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2, wkv_no_grad=True).to(device)
state = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = state.get('model', state.get('model_state_dict', state))
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k: del sd[k]
model.load_state_dict(sd, strict=False)
for p in model.parameters(): p.requires_grad = False
torch.cuda.empty_cache()
print(f"Base frozen. Adding LoRA (r={LORA_R})...")

# ── LoRA wrapper ──
class LoRALayer(nn.Module):
    def __init__(self, base, r, alpha=1.0):
        super().__init__()
        self.base = base
        out_dim, in_dim = base.weight.shape
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim))
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B)

# target module name prefixes
TARGET_PREFIXES = ['consolidate', 'tmix_r.', 'tmix_k.', 'tmix_v.', 'tmix_a.',
                   'experts.', 'router']

def attach_lora(module, path=''):
    for name, child in module.named_children():
        full = f"{path}.{name}" if path else name
        if isinstance(child, nn.Linear) and any(full.startswith(p) for p in TARGET_PREFIXES):
            setattr(module, name, LoRALayer(child, LORA_R).to(device))
        else:
            attach_lora(child, full)

attach_lora(model)
lora_params = [p for p in model.parameters() if p.requires_grad]
print(f"LoRA params: {sum(p.numel() for p in lora_params)/1e6:.2f}M")
model.train()
opt = torch.optim.AdamW(lora_params, lr=LR)
assistant_ids = tokenizer.encode("<|assistant|>\n")

# ── Pre-tokenize (with cache) ──
CACHE_TOK = f'experiments/checkpoints/sft_data/tokens_{SEQ}.npy'
CACHE_MASK = f'experiments/checkpoints/sft_data/masks_{SEQ}.npy'
CACHE_LEN = f'experiments/checkpoints/sft_data/lens_{SEQ}.npy'
if os.path.exists(CACHE_TOK):
    print("Loading cached tokenized data...")
    tok_arr = np.load(CACHE_TOK)
    mask_arr = np.load(CACHE_MASK)
    len_arr = np.load(CACHE_LEN)
    all_tokens = [tok_arr[i, :len_arr[i]].tolist() for i in range(len(len_arr))]
    all_masks = [mask_arr[i, :len_arr[i]].tolist() for i in range(len(len_arr))]
else:
    print("Pre-tokenizing...")
    with open(SFT_DATA, encoding='utf-8') as f:
        texts = [json.loads(l)["text"] for l in f]
    max_seq = SEQ
    tok_arr = np.zeros((len(texts), max_seq), dtype=np.int32)
    mask_arr = np.zeros((len(texts), max_seq), dtype=np.int32)
    len_arr = np.zeros(len(texts), dtype=np.int32)
    for i, text in tqdm(enumerate(texts), desc="Tokenize", total=len(texts)):
        tids = tokenizer.encode(text)[:SEQ]
        mask = [0] * len(tids)
        for j in range(len(tids) - len(assistant_ids) + 1):
            if tids[j:j+len(assistant_ids)] == assistant_ids:
                asst_end = j + len(assistant_ids)
                for k in range(asst_end, len(tids)): mask[k] = 1
                break
        tok_arr[i, :len(tids)] = tids
        mask_arr[i, :len(tids)] = mask
        len_arr[i] = len(tids)
    np.save(CACHE_TOK, tok_arr); np.save(CACHE_MASK, mask_arr); np.save(CACHE_LEN, len_arr)
    all_tokens = [tok_arr[i, :len_arr[i]].tolist() for i in range(len(len_arr))]
    all_masks = [mask_arr[i, :len_arr[i]].tolist() for i in range(len(len_arr))]
    print(f"Tokenized {len(texts)} examples, cached")

# ── Train ──
steps = 0
perm = torch.randperm(len(all_tokens))
for epoch in range(1):
    if steps >= N_STEPS: break
    torch.manual_seed(42 + epoch); perm = torch.randperm(len(all_tokens))
    pbar = tqdm(range(0, len(perm), BSZ), desc=f"SFT LoRA ep{epoch+1}")
    for start in pbar:
        if steps >= N_STEPS: break
        batch_i = perm[start:start+BSZ].tolist()
        max_len = max(len(all_tokens[i]) for i in batch_i)
        x = torch.zeros(len(batch_i), max_len, dtype=torch.long, device=device)
        lm = torch.zeros(len(batch_i), max_len, dtype=torch.long, device=device)
        for bi, i in enumerate(batch_i):
            tids = torch.tensor(all_tokens[i], dtype=torch.long)
            x[bi, :len(tids)] = tids
            lm[bi, :len(all_masks[i])] = torch.tensor(all_masks[i], dtype=torch.long)
        opt.zero_grad()
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1), reduction='none')
        loss = (loss * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        torch.cuda.empty_cache()
        ppl = float(torch.exp(loss))
        pbar.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{ppl:.1f}")
        steps += 1
        if steps % SAVE_EVERY == 0:
            torch.save({n: p.data for n, p in model.named_parameters() if p.requires_grad},
                       os.path.join(OUT_DIR, f"lora_{steps}.pt"))
    if steps >= N_STEPS: break

torch.save({n: p.data for n, p in model.named_parameters() if p.requires_grad},
           os.path.join(OUT_DIR, "lora_final.pt"))
print(f"Done: {OUT_DIR}/lora_final.pt")
