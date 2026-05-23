import os, shutil
# Check MSVC paths
for p in [
    r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
    r'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    r'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat',
]:
    print(f'{os.path.exists(p)}: {p}')
cl = shutil.which('cl.exe')
print(f'cl.exe in PATH: {cl}')
cuda = os.environ.get('CUDA_PATH', 'NO')
print(f'CUDA_PATH: {cuda}')
nvcc = shutil.which('nvcc')
print(f'nvcc in PATH: {nvcc}')
