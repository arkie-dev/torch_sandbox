class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[200, 100]", primals_2: "f32[200]", primals_3: "f32[200, 100]", primals_4: "f32[10, 200]", primals_5: "f32[10]"):
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        permute: "f32[100, 200]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[200, 200]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        relu: "f32[200, 200]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        permute_1: "f32[200, 10]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[200, 10]" = torch.ops.aten.addmm.default(primals_5, relu, permute_1);  primals_5 = permute_1 = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        amax: "f32[200, 1]" = torch.ops.aten.amax.default(addmm_1, [1], True)
        sub: "f32[200, 10]" = torch.ops.aten.sub.Tensor(addmm_1, amax);  addmm_1 = amax = None
        exp: "f32[200, 10]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[200, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[200, 10]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        return (div, primals_3, primals_4, relu, div)
        