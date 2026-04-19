import torch

a = torch.zeros(2, 3)  #all zeros
b = torch.ones(2, 3)   #all are one
c = torch.rand(2, 3)

print(a)
print(b)
print(c)

x = a + b
print(x)
print(x + c)

