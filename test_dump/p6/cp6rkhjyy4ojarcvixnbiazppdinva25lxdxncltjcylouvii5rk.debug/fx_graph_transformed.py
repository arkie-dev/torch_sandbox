class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[200, 100]", primals_2: "f32[200]", primals_3: "f32[200, 100]", primals_4: "f32[10, 200]", primals_5: "f32[10]"):
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        permute: "f32[100, 200]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[200, 200]" = torch.ops.aten.mm.default(primals_3, permute);  permute = None
        add_tensor: "f32[200, 200]" = torch.ops.aten.add.Tensor(mm_default, primals_2);  mm_default = primals_2 = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        relu: "f32[200, 200]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        permute_1: "f32[200, 10]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[200, 10]" = torch.ops.aten.addmm.default(primals_5, relu, permute_1);  primals_5 = permute_1 = None
        
        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(addmm_1, 1)
        getitem: "f32[200, 1]" = prepare_softmax_online_default[0]
        getitem_1: "f32[200, 1]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[200, 10]" = torch.ops.aten.sub.Tensor(addmm_1, getitem);  addmm_1 = getitem = None
        exp_default: "f32[200, 10]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        
         # File: /home/devro/workspace/main/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
        div: "f32[200, 10]" = torch.ops.aten.div.Tensor(exp_default, getitem_1);  exp_default = getitem_1 = None
        return (div, primals_3, primals_4, relu, div)
        