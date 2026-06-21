#!/usr/bin/env bash
# RINA — 完整评估脚本
# 复现 Gen 5 所有实验的生成和消融对比
# 用法: bash eval_all.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo " RINA Evaluation Suite"
echo "========================================"
echo ""

# ── 1. Baseline —— out-final（原始 MLA, fp32）──
echo "=== 1. Baseline (out-final, fp32) ==="
python3 -c "
import torch
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_A_Config(vocab_size=50257, block_size=512)
m = RINA_A(cfg).to(device); m.eval()
sd = torch.load('models/out-final/rina-gen5-baseline-fp32.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            l,_,_ = m(x)
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

# ── 2. int4 only (out-quant) ──
echo "=== 2. int4 only (out-quant) ==="
python3 -c "
import torch
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_A_Config(vocab_size=50257, block_size=512, use_int4=True)
m = RINA_A(cfg).to(device); m.eval()
sd = torch.load('models/out-quant/rina-gen5-baseline-int4.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            l,_,_ = m(x)
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

# ── 3. Route A v3 (Latent Indexed Attention, full attn + int4) ──
echo "=== 3. Route A v3 (full attn, int4, triplet contrastive) ==="
python3 -c "
import torch
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_A_Config(vocab_size=50257, block_size=512, use_int4=True)
m = RINA_A(cfg).to(device); m.eval()
sd = torch.load('models/out-rina-a-v3/rina-gen5-route-a-v3.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            l,_,_ = m(x)
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

# ── 4. Route C (Inertia Wave, no attention) ──
echo "=== 4. Route C (Inertia Wave, no attention) ==="
python3 -c "
import torch
from rina.model_c import RINA_C, RINA_C_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_C_Config(vocab_size=50257, block_size=512)
m = RINA_C(cfg).to(device); m.eval()
sd = torch.load('models/out-rina-c/rina-gen5-route-c.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            l,_ = m(x)
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

# ── 5. AC Hybrid (inertia + attention + sparse) ──
echo "=== 5. AC Hybrid (inertia+attn+sparse) ==="
python3 -c "
import torch
from rina.model_ac import RINA_AC, RINA_AC_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_AC_Config(vocab_size=50257, block_size=512, use_int4=True, sparse_k=8, sparse_local_w=4)
m = RINA_AC(cfg).to(device); m.eval()
sd = torch.load('models/out-rina-ac/rina-gen5-route-ac-hybrid.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            out = m(x); l = out[0]
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

# ── 6. Ablation: AC hybrid with inertia layers zeroed ──
echo "=== 6. Ablation: AC hybrid (inertia layers zeroed) ==="
python3 -c "
import torch
from rina.model_ac import RINA_AC, RINA_AC_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg = RINA_AC_Config(vocab_size=50257, block_size=512, use_int4=True, sparse_k=8, sparse_local_w=4)
m = RINA_AC(cfg).to(device); m.eval()
sd = torch.load('models/out-rina-ac/rina-gen5-route-ac-hybrid.pt', map_location=device, weights_only=False)['model']
m.load_state_dict(sd, strict=False)
for n, p in m.named_parameters():
    if any(f'h.{i}.' in n for i in [0,1,2,3]):
        p.data.zero_()

for prompt in ['The capital of France is', 'Alice and Bob', 'In the beginning']:
    ids=tok.encode(prompt)[:16]; x=torch.tensor([ids],device=device)
    with torch.no_grad():
        for _ in range(25):
            out = m(x); l = out[0]
            p=F.softmax(l[:,-1].float()/1.0,-1);p[0,0]=0
            x=torch.cat([x,torch.multinomial(p,1)],1)
    print(f'  {prompt}: {tok.decode(x[0].tolist())}')
"
echo ""

echo "========================================"
echo " Evaluation complete."
echo " Compare results with docs/RINA实验日志.md"
echo "========================================"
