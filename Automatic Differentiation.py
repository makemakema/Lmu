import torch
x = torch.arange(4.0)
print(x)

x.requires_grad_(True) #等价于x = torch.arange(4.0, requires_grad=Ture)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad, y)

x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad, y)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad==u, y, u, z, x.grad)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(d, a, a.grad, d/a)

print(a.grad == d/a)