import torch
from torch import nn

torch.device('cpu')
torch.device('cuda')
torch.device('cuda:1')


print(torch.cuda.device_count())


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu()
try_gpu(10)
try_all_gpus()

# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())

X = torch.tensor([1, 2, 3])
print(X.device)

x = torch.ones(2, 3, device=try_gpu())
print(x.device)
print(x)

# 至少两张gpu卡才能尝试
# Z = X.cuda(1)
# print(X)
# print(Z)

Z = x.to('cuda')
print(Z)

print(x + Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(x))

print(net[0].weight.data.device)