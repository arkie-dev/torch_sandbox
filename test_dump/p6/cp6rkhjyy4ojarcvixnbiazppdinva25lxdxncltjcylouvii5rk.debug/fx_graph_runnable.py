
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = '+all'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'test_dump'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
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



isolate_fails_code_str = None




# torch version: 2.8.0+cu128
# torch cuda version: 12.8
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 5070 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_5, relu, permute_1);  primals_5 = permute_1 = None
        amax = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        return (div, primals_3, primals_4, relu, div)
        
def load_args(reader):
    buf0 = reader.storage(None, 80000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (200, 100), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 800, device=device(type='cuda', index=0))
    reader.tensor(buf1, (200,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 80000, device=device(type='cuda', index=0))
    reader.tensor(buf2, (200, 100), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 8000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (10, 200), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 40, device=device(type='cuda', index=0))
    reader.tensor(buf4, (10,), is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)