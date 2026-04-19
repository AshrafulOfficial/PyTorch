import torch

#crete torch

#tensor is a container of numbers of ML

#scalar
x = torch.tensor(5)
print(x)

#vector
y = torch.tensor([1, 2, 3])
print(y)

#matrix
z = torch.tensor([[1, 2], [3, 4]])
print(z)

print(y.shape)
print(z.shape)

#size 3 means: 3 elements
#size 2 2 means: 2 rows, 2 columns