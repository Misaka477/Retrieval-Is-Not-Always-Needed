@echo off
echo ========================================
echo  RINA — Reproduce Environment
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

:: 2. Quick smoke test
echo [2/3] Running smoke test...
python scripts/quick_test.py
if %errorlevel% neq 0 (
    echo ERROR: smoke test failed
    pause
    exit /b 1
)
echo   OK
echo.

:: 3. Model info
echo [3/3] Environment info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo ========================================
echo  All checks passed.
echo  Next: python scripts/train.py
echo ========================================
pause
