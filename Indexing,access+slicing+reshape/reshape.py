import torch
x = torch.rand(2, 3)

print(x.shape)

#now reshape

y = x.view(3, 2)
print(y.shape)