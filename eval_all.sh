#!/usr/bin/env bash
# RINA — 完整评估脚本
# 复现 Gen 5 + Gen 6 Jamba 实验的生成和消融对比
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "   RINA Evaluation Suite"
echo "========================================"

# ── 1. Jamba v1 (q4+q2 KV baseline) ──
echo ""
echo "=== 1. Jamba v1 (q4+q2 KV, CE 4.8) ==="
python3 -c "
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from rina.model_jamba import RINA_Jamba, RJ_Config
device='cuda'
tok=AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token=tok.eos_token
cfg=RJ_Config(vocab_size=128256,block_size=512,use_int4=True,
    n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
    sparse_k=16,sparse_window=32,sparse_local_w=4,ssm_steps=3)
m=RINA_Jamba(cfg).to(device).eval()
sd=torch.load('models/out-rina-jamba-v1/jamba_final.pt',map_location=device,weights_only=False)
m.load_state_dict(sd['model'],strict=False)
for pr in ['The capital of France is','The meaning of life is','Once upon a time,','In the theory of relativity,']:
    ids=tok.encode(pr,return_tensors='pt').to(device)
    x=ids.clone();g=set()
    for _ in range(64):
        with torch.no_grad(): l,_=m(x)
        lgt=l[0,-1].float()/0.3
        vv,_=torch.topk(lgt,20);lgt[lgt<vv[-1]]=-float('Inf')
        p=F.softmax(lgt,-1)
        for gid in list(g):
            if gid<p.size(-1):p[gid]/=5.0
        p=p/p.sum()
        nxt=torch.multinomial(p.unsqueeze(0),1);g.add(nxt.item());x=torch.cat([x,nxt],1)
    print(f'  {pr}: {tok.decode(x[0].tolist())[:150]}')
"

# ── 2. Jamba v2 (q2+q1 3-bit KV) ──
echo ""
echo "=== 2. Jamba v2 (q2+q1 3-bit KV, CE 4.2) ==="
python3 -c "
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from rina.model_jamba_lq import RINA_Jamba_LQ, RJLQ_Config
device='cuda'
tok=AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token=tok.eos_token
cfg=RJLQ_Config(vocab_size=128256,block_size=512,use_int4=True,
    n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
    sparse_k=16,sparse_window=32,sparse_local_w=4,ssm_steps=3,
    quant_mode='q2k_q1v',ssm_qbits=0)
m=RINA_Jamba_LQ(cfg).to(device).eval()
sd=torch.load('models/out-rina-jamba-v2/jamba_final.pt',map_location=device,weights_only=False)
m.load_state_dict(sd['model'],strict=False)
for pr in ['The capital of France is','The meaning of life is','Once upon a time,','In the theory of relativity,']:
    ids=tok.encode(pr,return_tensors='pt').to(device)
    x=ids.clone();g=set()
    for _ in range(64):
        with torch.no_grad(): l,_=m(x)
        lgt=l[0,-1].float()/0.3
        vv,_=torch.topk(lgt,20);lgt[lgt<vv[-1]]=-float('Inf')
        p=F.softmax(lgt,-1)
        for gid in list(g):
            if gid<p.size(-1):p[gid]/=5.0
        p=p/p.sum()
        nxt=torch.multinomial(p.unsqueeze(0),1);g.add(nxt.item());x=torch.cat([x,nxt],1)
    print(f'  {pr}: {tok.decode(x[0].tolist())[:150]}')
"

# ── 3. Jamba LSC q4 (log-space SSM) ──
echo ""
echo "=== 3. Jamba LSC q4 (log-space SSM, q4 intermediates) ==="
python3 -c "
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from rina.model_jamba_lq import RINA_Jamba_LQ, RJLQ_Config
device='cuda'
tok=AutoTokenizer.from_pretrained('models/teacher/LLM-Research/Llama-3___2-1B-Instruct')
tok.pad_token=tok.eos_token
cfg=RJLQ_Config(vocab_size=128256,block_size=512,use_int4=True,
    n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
    sparse_k=16,sparse_window=32,sparse_local_w=4,ssm_steps=3,
    quant_mode='q4k_q2v',ssm_qbits=0)
m=RINA_Jamba_LQ(cfg).to(device).eval()
sd=torch.load('models/out-rina-jamba-lq-q4/jambalq_final.pt',map_location=device,weights_only=False)
m.load_state_dict(sd['model'],strict=False)
for pr in ['The capital of France is','The meaning of life is','Once upon a time,','In the theory of relativity,']:
    ids=tok.encode(pr,return_tensors='pt').to(device)
    x=ids.clone();g=set()
    for _ in range(64):
        with torch.no_grad(): l,_=m(x)
        lgt=l[0,-1].float()/0.3
        vv,_=torch.topk(lgt,20);lgt[lgt<vv[-1]]=-float('Inf')
        p=F.softmax(lgt,-1)
        for gid in list(g):
            if gid<p.size(-1):p[gid]/=5.0
        p=p/p.sum()
        nxt=torch.multinomial(p.unsqueeze(0),1);g.add(nxt.item());x=torch.cat([x,nxt],1)
    print(f'  {pr}: {tok.decode(x[0].tolist())[:150]}')
"

echo ""
echo "========================================"
echo " Evaluation complete."
echo " See docs/RINA_实验总览.md for results."
echo "========================================"
