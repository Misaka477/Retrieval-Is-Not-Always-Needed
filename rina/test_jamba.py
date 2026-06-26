#!/usr/bin/env python3
"""Jamba 快速生成测试：比较 v1/v2/LSC q4 三个版本"""
import torch, torch.nn.functional as F, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token = tok.eos_token

from rina.model_jamba import RINA_Jamba, RJ_Config
from rina.model_jamba_lq import RINA_Jamba_LQ, RJLQ_Config

def gen(model, pr, temp=0.3, top_k=20, rep=5.0, steps=48):
    ids = tok.encode(pr, return_tensors='pt').to(device)
    x = ids.clone(); gen_ids = set()
    for _ in range(steps):
        with torch.no_grad():
            l, _ = model(x)
        logits = l[0, -1].float() / temp
        if top_k > 0:
            vv, _ = torch.topk(logits, top_k)
            logits[logits < vv[-1]] = -float('Inf')
        p = F.softmax(logits, -1)
        if rep > 1.0 and gen_ids:
            for gid in list(gen_ids):
                if gid < p.size(-1): p[gid] /= rep
            p = p / p.sum()
        nxt = torch.multinomial(p.unsqueeze(0), 1)
        gen_ids.add(nxt.item()); x = torch.cat([x, nxt], 1)
    return tok.decode(x[0].tolist())

prompts = ['The capital of France is', 'The meaning of life is', 'Once upon a time,']

models = []

# v1: Jamba baseline (q4+q2 KV)
print("Loading v1 (baseline, q4+q2 KV)...")
cfg1 = RJ_Config(vocab_size=128256, block_size=512, use_int4=True,
    n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
    sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3)
m1 = RINA_Jamba(cfg1).to(device).eval()
sd1 = torch.load('models/out-rina-jamba-v1/jamba_final.pt', map_location=device, weights_only=False)
m1.load_state_dict(sd1['model'], strict=False)
models.append(('v1 (q4+q2 KV, CE 4.8)', m1))

# v2: q2+q1 KV (3-bit KV cache)
print("Loading v2 (3-bit KV, q2+q1)...")
cfg2 = RJ_Config(vocab_size=128256, block_size=512, use_int4=True,
    n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
    sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3)
m2 = RINA_Jamba(cfg2).to(device).eval()
sd2 = torch.load('models/out-rina-jamba-v2/jamba_final.pt', map_location=device, weights_only=False)
m2.load_state_dict(sd2['model'], strict=False)
models.append(('v2 (3-bit KV, CE 4.2)', m2))

# LSC q4: log-space SSM + q4 intermediates
print("Loading LSC q4 (log-space SSM, q4)...")
cfg3 = RJLQ_Config(vocab_size=128256, block_size=512, use_int4=True,
    n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
    sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3,
    quant_mode='q4k_q2v', ssm_qbits=0)
m3 = RINA_Jamba_LQ(cfg3).to(device).eval()
sd3 = torch.load('models/out-rina-jamba-lq-q4/jambalq_final.pt', map_location=device, weights_only=False)
m3.load_state_dict(sd3['model'], strict=False)
models.append(('LSC q4 (log-space SSM, CE ~5.7)', m3))

for label, m in models:
    print(f'\n{"="*60}')
    print(f' {label}')
    print(f'{"="*60}')
    for pr in prompts:
        out = gen(m, pr)
        print(f'  P: {pr}')
        print(f'  G: {out[:120]}')
        print()
