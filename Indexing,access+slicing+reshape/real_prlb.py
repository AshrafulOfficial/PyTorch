import torch

img = torch.rand(3, 224, 224)
print(img.shape)
print(img[0].shape)