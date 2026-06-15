@echo off
chcp 65001 >nul
echo ========================================
echo  RINA — Retrieval Is Not Always Needed
echo  Setup for Windows
echo ========================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)
echo   OK
echo.

echo [2/3] Setting up GPT-2 tokenizer...
python -c "from transformers import GPT2Tokenizer; import os; os.makedirs('checkpoints/gpt2_tokenizer', exist_ok=True); tok = GPT2Tokenizer.from_pretrained('gpt2'); tok.save_pretrained('checkpoints/gpt2_tokenizer')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: GPT-2 tokenizer download failed, run manually: python -c ^"from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2').save_pretrained('checkpoints/gpt2_tokenizer')^"
) else (
    echo GPT-2 tokenizer: ready
)
echo.

echo [3/3] Testing model...
python -c "import torch; from rina import RINA, RINAConfig; c=RINAConfig(vocab_size=50257,n_layer=4); m=RINA(c); x=torch.randint(0,50257,(1,16)); l,_=m(x,x); print(f'Forward OK: loss={l.item():.2f}')"
if %errorlevel% neq 0 (
    echo ERROR: model test failed
    pause
    exit /b 1
)
echo   OK
echo.

echo ========================================
echo  Ready. Train model: python rina/train.py
echo ========================================
pause
