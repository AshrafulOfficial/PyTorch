#it takes partial value from matrix

import torch

a = torch.tensor([[10, 20, 30],
                    [40, 50, 60]])

print(a[0])
print(a[1])

#Take multiple row value

print(a[:, 2]) #take all 2 clumns value from all rows
print(a[1, :])  #row 1, all column
print(a[0:2, :])  #first 2 rows