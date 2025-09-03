import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

torch.set_default_device('cuda')

model = Model()

compiled_model = torch.compile(model, backend='inductor')


print(compiled_model(torch.randn(200, 100)) )