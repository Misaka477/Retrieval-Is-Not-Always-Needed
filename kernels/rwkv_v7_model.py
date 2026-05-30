########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''
try:
    print('RWKV_KERNEL', os.environ["RWKV_KERNEL"])
except:
    os.environ["RWKV_KERNEL"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    assert HEAD_SIZE == 64 # can change 64 to your HEAD_SIZE

    # check https://github.com/BlinkDL/RWKV-CUDA/blob/main/rwkv7_fast_fused/rwkv7_cuda_benchmark.py
    #
    # use rwkv7_clampw_v3.cpp and rwkv7_clampw_v3_for_h100.cu for 20% faster fwd & bwd kernel on H100s and some consumer GPUS (for some Bsz*Headcount)
    # note: sometimes rwkv7_clampw_v3_for_h100_alt is faster

    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    if "@rwkv3" in os.environ["RWKV_KERNEL"]:
        RWKV7_CLAMPW_OP = torch.ops.rwkv7_clampw_v3
        load(name="rwkv7_clampw_v3", sources=['cuda/rwkv7_clampw_v3_for_h100.cu', 'cuda/rwkv7_clampw_v3.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    else:
        RWKV7_CLAMPW_OP = torch.ops.rwkv7_clampw
        load(name="rwkv7_clampw", sources=['cuda/rwkv7_clampw.cu', 'cuda/rwkv7_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx,r,w,k,v,a,b):
            B,T,H,N = r.shape
            assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
            assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,N,N, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,N, dtype=torch.float32,device=w.device)
            RWKV7_CLAMPW_OP.forward(r,w,k,v,a,b,y,s,sa)
            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
            return y
        @staticmethod
        def backward(ctx,dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r,w,k,v,a,b,s,sa = ctx.saved_tensors
            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
            RWKV7_CLAMPW_OP.backward(r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db)
            return dr,dw,dk,dv,da,db
    def RWKV7_CLAMPW_CUDA(r,w,k,v,a,b):
        B,T,HN = r.shape
        r,w,k,v,a,b = [i.view(B,T,HN//64,64) for i in [r,w,k,v,a,b]] # can change 64 to your HEAD_SIZE. have to hard-code the number here, or pytorch will complain
        return RWKV7_CLAMPW_CUDA_OP.apply(r,w,k,v,a,b).view(B,T,HN)

########################################################################################################

load(name="rwkv7_cmix_bf16_v5", sources=["cuda/rwkv7_cmix_bf16_v5.cpp","cuda/rwkv7_cmix_bf16_v5.cu"], extra_cflags=["-O3"],
     extra_cuda_cflags=['-res-usage', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"],
     is_python_module=False, verbose=True)

class _CmixLayerV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        out, mixed, act = torch.ops.rwkv7_cmix_bf16_v5.forward(
            x.contiguous(),
            x_k.contiguous(),
            key_weight.contiguous(),
            value_weight.contiguous(),
        )
        ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, act)

# ... truncated
