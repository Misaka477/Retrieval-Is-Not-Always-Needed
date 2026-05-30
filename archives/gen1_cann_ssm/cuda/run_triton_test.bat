@echo off
call "D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set CC="D:\Software_Development\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
python -c "import sys; sys.path.insert(0,'D:\\Software_Development\\Project\\RINA_Project'); from modules.cann_triton import test; test()"
