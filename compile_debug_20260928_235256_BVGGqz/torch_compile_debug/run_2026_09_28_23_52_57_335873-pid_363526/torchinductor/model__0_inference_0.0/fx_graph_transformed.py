class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[65537]"):
        # File: /home/arki/workspace/pytorch/agent_space/extern_kernel_out_unaligned.py:37 in fn, code: x = storage[1:]
        slice_1: "f32[65536]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, 9223372036854775807);  arg0_1 = None

        # File: /home/arki/workspace/pytorch/agent_space/extern_kernel_out_unaligned.py:38 in fn, code: y = torch.ops.alignment_demo.copy(x)  # Lowers to ExternKernelOut.
        copy: "f32[65536]" = torch.ops.alignment_demo.copy.default(slice_1);  slice_1 = None

        # File: /home/arki/workspace/pytorch/agent_space/extern_kernel_out_unaligned.py:39 in fn, code: return y + 1  # Inductor generates the Triton consumer.
        add: "f32[65536]" = torch.ops.aten.add.Tensor(copy, 1);  copy = None
        return (add,)
