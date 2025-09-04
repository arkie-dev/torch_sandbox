
import os
os.environ['TRITON_DUMP_DIR'] = '/home/arki/workspace/pytorch/agent_space/compile_debug_20260928_235256_BVGGqz/triton'
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/home/arki/workspace/pytorch/agent_space/compile_debug_20260928_235256_BVGGqz'
os.environ['TRITON_KERNEL_DUMP'] = '1'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_arki/tmpwqjwaw2j'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_arki/tmpwqjwaw2j/triton'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.15.0.dev20260922+cu132
# torch cuda version: 13.2
# torch git version: 038331dffa75e17cc9381edd0c2c667f07adc2a7


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Aug_15_22:02:13_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.140 
# Build cuda_12.2.r12.2/compiler.33191640_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 5070 Ti : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, 9223372036854775807);  arg0_1 = None
        copy = torch.ops.alignment_demo.copy.default(slice_1);  slice_1 = None
        add = torch.ops.aten.add.Tensor(copy, 1);  copy = None
        return (add,)

def load_args(reader):
    buf0 = reader.storage(None, 262148, device=device(type='cuda', index=0))
    reader.tensor(buf0, (65537,), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # mod(*args)