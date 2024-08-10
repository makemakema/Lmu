import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print(net)
print(net[0].weight)

X = torch.rand(2, 20)
net(X)

print(net)
print(net[0].weight.shape)
print(net[2].weight.shape)

#undetermined
@d2l.add_to_class(d2l.Module)
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)