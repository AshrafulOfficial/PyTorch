import torch

a = torch.tensor([[10, 20, 30],
                    [40, 50, 60]])

print(a[0][0], a[0][1], a[0][2])   #a[row][column]
print(a[1][0], a[1][1], a[1][2])