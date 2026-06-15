#!/bin/bash
set -e

echo "========================================"
echo " RINA — Retrieval Is Not Always Needed"
echo " Setup for Linux"
echo "========================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt
echo "  OK"
echo ""

echo "[2/3] Downloading GPT-2 tokenizer..."
python3 -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('checkpoints/gpt2_tokenizer')" 2>/dev/null || \
python3 -c "
from transformers import GPT2Tokenizer
import os; os.makedirs('checkpoints/gpt2_tokenizer', exist_ok=True)
tok = GPT2Tokenizer.from_pretrained('gpt2')
tok.save_pretrained('checkpoints/gpt2_tokenizer')
print('GPT-2 tokenizer saved')
"
echo "  OK"
echo ""

echo "[3/3] Testing model..."
python3 -c "import torch; from rina import RINA, RINAConfig; c=RINAConfig(vocab_size=50257,n_layer=4); m=RINA(c); x=torch.randint(0,50257,(1,16)); l,_=m(x,x); print(f'Forward OK: loss={l.item():.2f}')"
echo "  OK"
echo ""

echo "========================================"
echo " Ready."
echo " Train: python3 rina/train.py"
echo "========================================"
