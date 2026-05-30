@echo off
call "D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;%PATH%

echo === Compiling cann_step.cu ===
nvcc -shared -o D:\Software_Development\Project\RINA_Project\modules\cann_step.dll ^
     D:\Software_Development\Project\RINA_Project\modules\cann_step.cu ^
     -I"D:\Software_Development\CUDA_Toolkit_12.4\include" ^
     -D_GNU_SOURCE --compiler-options=-MD -O3 ^
     2>&1
if %ERRORLEVEL% EQU 0 (
    echo COMPILE SUCCESS
) else (
    echo COMPILE FAILED
)
