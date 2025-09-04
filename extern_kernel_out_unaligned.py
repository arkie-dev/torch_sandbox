import torch
from torch._inductor import config


UNALIGNED_OUT = True  # Set False for the aligned control; run in a fresh process.

lib = torch.library.Library("alignment_demo", "DEF")
lib.define("copy(Tensor x) -> Tensor")
lib.define(
    "copy.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)",
    tags=(torch.Tag.out,),
)
lib.impl("copy", lambda x: x.clone(), "CUDA")


@torch.library.impl(lib, "copy.out", "CUDA")
def copy_out(x, *, out):
    print(f"input pointer % 16: {x.data_ptr() % 16}", flush=True)
    before = out.data_ptr() % 16

    storage = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)
    out.set_(storage[1:])
    
    print(f"out pointer % 16: {before} -> {out.data_ptr() % 16}", flush=True)
    out.copy_(x)
    return out


@torch.library.register_fake("alignment_demo::copy")
def copy_fake(x):
    return torch.empty_like(x)


@torch.compile(fullgraph=True)
def fn(storage):
    # Slice inside the graph so the custom op has an unaligned IR input.
    x = storage[1:]
    y = torch.ops.alignment_demo.copy(x)  # Lowers to ExternKernelOut.
    return y + 1  # Inductor generates the Triton consumer.


if __name__ == "__main__":
    storage = torch.randn(65537, device="cuda", dtype=torch.float32)
    expected = storage[1:] + 1

    actual = fn(storage)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    print("Output matches; no alignment failure observed on this run.")
