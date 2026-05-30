@echo off
echo ========================================
echo  MoHE-RWKV — Reproduce Environment
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
echo [2/3] Compiling WKV kernel and verifying model...

python -c "
import torch
from rina import MoHERWKV
m = MoHERWKV(65536, 768, 1536, n_experts=12).cuda()
x = torch.randint(0, 65536, (1, 64), device='cuda')
with torch.no_grad():
    y = m(x)
    print(f'Model forward OK: logits shape {list(y.shape)}')
    print(f'Params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
"
if %errorlevel% neq 0 (
    echo ERROR: model test failed
    pause
    exit /b 1
)
echo   OK
echo.

:: 3. Check data availability
echo [3/3] Checking data...
if exist checkpoints/mohe_fw_rwkv.npy (
    python -c "import numpy as np; d=np.load('checkpoints/mohe_fw_rwkv.npy',mmap_mode='r'); print(f'Training data: {len(d):,} tokens')"
) else (
    echo Data not found. Run: python experiments/prepare_data_rwkv.py
)
if exist checkpoints/mohe_transferred_init.pt (
    echo Init weights: found
) else (
    echo Init weights not found. Run: python experiments/weight_transfer.py
)
echo.
echo ========================================
echo  Environment ready.
echo  Train: python experiments/mohe_transferred_train.py
echo  Generate: python experiments/generate.py
echo ========================================
pause
