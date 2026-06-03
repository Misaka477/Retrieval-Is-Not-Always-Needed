"""Test: does real Hebbian pattern update at inference break 'is is is'?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import torch
from rina.model import WKV7Fn, _load_wkv7
from rina import MoHERWKV_V5
from rina.rwkv_tokenizer import TRIE_TOKENIZER

_load_wkv7()
device = 'cuda'
VOCAB, DM, NP = 65536, 768, 3072
SEQ = 512; HEB_LR = 0.5; DECAY = 0.99

tok = TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
model = MoHERWKV_V5(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
ckpt = torch.load('checkpoints/mohe_v5_phase2.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

def hebbian_forward(model, x, patterns, do_hebbian):
    """Run model forward, optionally update patterns with Hebbian outer product."""
    B = x.shape[0]
    emb = model.embed_norm(model.embed(x))
    B, T, D = emb.shape; H, N = D // 64, 64

    w = torch.exp(-torch.exp(model.tmix_w))
    w4d = w.unsqueeze(0).unsqueeze(0).expand(B, T, H, N).contiguous()
    r = model.tmix_r(emb).view(B, T, H, N).contiguous()
    k = model.tmix_k(emb).view(B, T, H, N).contiguous()
    v = model.tmix_v(emb).view(B, T, H, N).contiguous()
    a = model.tmix_a(emb).view(B, T, H, N).contiguous() * 0.01
    h = WKV7Fn.apply(r, w4d, k, v, -a, a.clone()).view(B, T, D)

    # Hebbian: update patterns via outer product (h drives pattern rotation)
    if do_hebbian and h.shape[1] > 1:
        h_avg = h.mean(dim=1)  # [B, D]
        h_global = h_avg.mean(dim=0, keepdim=True)  # [1, D]
        for p in patterns:
            scores = torch.relu(h @ p.T)  # [B, T, 3072]
            s_avg = scores.mean(dim=[0, 1])  # [3072]
            delta = (s_avg.unsqueeze(-1) @ h_global).squeeze(0)  # -> [3072, D]
            p.data.copy_(p * DECAY + delta * HEB_LR * (1 - DECAY))

    # depth loop (same as model.forward)
    for depth in range(3):
        route_raw = (model.router(h) + model.router_bias) * 3.0
        route_weights = torch.softmax(route_raw, dim=-1)
        h_exps = torch.stack([e(h, emb)[0] for e in model.experts], dim=0)
        h_exps = model.expert_norm(h_exps.permute(1,2,0,3).reshape(-1, D)).reshape(B, T, model.n_experts, D)
        if model.topk > 0 and model.topk < model.n_experts:
            _, inds = route_weights.topk(model.topk, dim=-1)
            mask = torch.zeros(B, T, model.n_experts, device=device).scatter_(-1, inds, 1)
            h_exps = h_exps * mask.unsqueeze(-1)
        h = model.consolidate_norm(model.consolidate(h_exps.reshape(B, T, model.n_experts * D)))
    logits = model.head(h)
    return logits

prompts = ['The meaning of life is', 'Hello world']
orig = [e.patterns.data.clone() for e in model.experts]

for mode, do_heb in [('Baseline (no Hebbian)', False), ('With Hebbian', True)]:
    print(f'\n=== {mode} ===')
    for prompt in prompts:
        # restore original patterns
        for e, o in zip(model.experts, orig): e.patterns.data.copy_(o)

        gen = tok.encode(prompt)
        with torch.no_grad():
            for _ in range(60):
                inp = torch.tensor([gen[-SEQ:]], device=device)
                logits = hebbian_forward(model, inp, [e.patterns for e in model.experts], do_heb)
                tid = torch.argmax(logits[0, -1]).item()
                if tid == 0: break
                gen.append(tid)
        text = ''.join(c if ord(c)<128 else '?' for c in tok.decodeBytes(gen).decode('utf-8', errors='replace'))
        print(f'  {prompt}: {text[:150]}')

# restore
for e, o in zip(model.experts, orig): e.patterns.data.copy_(o)
print('\nDone.')
