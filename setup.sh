#!/bin/bash
set -e

echo "========================================"
echo " RINA — Retrieval Is Not Always Needed"
echo " Gen 6 — Jamba Hybrid Setup (Linux)"
echo "========================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt
echo "  OK"
echo ""

echo "[2/3] Setting up Llama 3.2 tokenizer..."
python3 -c "
from transformers import AutoTokenizer
import os
cache = os.path.expanduser('~/.cache/huggingface')
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3___2-1B-Instruct',
    cache_dir=cache, token=True)
tok.pad_token = tok.eos_token
print(f'Tokenizer OK: vocab={tok.vocab_size}')
"
echo "  OK"
echo ""

echo "[3/3] Testing Jamba model..."
python3 -c "
import torch
from rina.model_jamba import RINA_Jamba, RJ_Config

cfg = RJ_Config(vocab_size=128256, block_size=512, use_int4=True,
    n_embd=640, n_layer=4, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
    sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3)
m = RINA_Jamba(cfg)
x = torch.randint(0, 128256, (1, 16))
l, ce = m(x[:, :-1], x[:, 1:])
print(f'Forward OK: logits={l.shape} ce={ce.item():.2f}')
"
echo "  OK"
echo ""

echo "========================================"
echo " Ready."
echo " Train: python3 -u rina/train_jamba.py --steps 50000"
echo "========================================"
