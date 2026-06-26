@echo off
chcp 65001 >nul
echo ========================================
echo  RINA — Retrieval Is Not Always Needed
echo  Gen 6 — Jamba Hybrid Setup (Windows)
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

echo [2/3] Setting up Llama 3.2 tokenizer...
python -c "from transformers import AutoTokenizer; import os; tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3___2-1B-Instruct', cache_dir=os.path.expanduser('~/.cache/huggingface'), token=True); tok.pad_token = tok.eos_token; print(f\"Tokenizer OK: vocab={tok.vocab_size}\")"
if %errorlevel% neq 0 (
    echo ERROR: Tokenizer download failed. Run manually:
    echo   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3___2-1B-Instruct', token=True)"
    pause
    exit /b 1
)
echo.
if %errorlevel% neq 0 (
    echo WARNING: GPT-2 tokenizer download failed, run manually
) else (
    echo   OK
)
echo.

echo [3/3] Testing Jamba model...
python -c "import torch; from rina.model_jamba import RINA_Jamba, RJ_Config; cfg = RJ_Config(vocab_size=128256, block_size=512, use_int4=True, n_embd=640, n_layer=4, n_head=10, n_kv_heads=5, d_c=160, head_dim=64, sparse_k=16, sparse_window=32, sparse_local_w=4, ssm_steps=3); m = RINA_Jamba(cfg); x = torch.randint(0, 128256, (1, 16)); l, ce = m(x[:, :-1], x[:, 1:]); print(f'Forward OK: logits={l.shape} ce={ce.item():.2f}')"
if %errorlevel% neq 0 (
    echo ERROR: model test failed
    pause
    exit /b 1
)
echo   OK
echo.

echo ========================================
echo  Ready.
echo  Train: python -u rina\train_jamba.py --steps 50000
echo ========================================
pause
