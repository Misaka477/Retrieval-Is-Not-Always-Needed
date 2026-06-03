@echo off
chcp 65001 >nul
echo ========================================
echo  RINA v3 — AR + Stateful Denoiser
echo  Requires: RTX 3070 Ti 8GB+ (CUDA 12+)
echo ========================================
echo.

:: 1. Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)
echo   OK
echo.

:: 2. Verify CUDA + compile WKV kernel
echo [2/3] Compiling WKV kernel and verifying backbone...
python -c "
import torch, os, sys
sys.path.insert(0, 'rina')
from rwkv_v7_demo import RWKV, args
m = RWKV(args).cuda()
x = torch.randint(0, 65536, (1, 64), device='cuda')
with torch.no_grad():
    y, h = m(x, return_h=True)
    print(f'Backbone forward OK: logits {list(y.shape)}, h {list(h.shape)}')
    print(f'Params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
"
if %errorlevel% neq 0 (
    echo ERROR: backbone test failed
    pause
    exit /b 1
)
echo   OK
echo.

:: 3. Check data
echo [3/3] Checking data...
if exist checkpoints\mohe_fw_rwkv_1b.npy (
    python -c "import numpy as np; d=np.load('checkpoints/mohe_fw_rwkv_1b.npy',mmap_mode='r'); print(f'Training data: {len(d):,} tokens')"
) else (
    echo WARNING: mohe_fw_rwkv_1b.npy not found in checkpoints/
)
if exist checkpoints\rwkv_vocab_v20230424.txt (
    echo Tokenizer: found
) else (
    echo WARNING: rwkv_vocab_v20230424.txt not found
)
if exist rwkv7-g1d-0.1b-20260129-ctx8192.pth (
    echo Backbone weights: found
) else (
    echo WARNING: backbone weights not found
)
echo.
echo ========================================
echo  Environment ready.
echo  Train denoiser: python rina/train_ar.py
echo  Train conf:     python rina/train_conf.py
echo  Evaluate:       python rina/eval_multi.py
echo ========================================
pause
