import torch
import torch.nn as nn


# A
x = torch.randn(4096, 4096)
y = torch.randn(192, 4096, 1)
z = torch.matmul(x, y)


# B
x = torch.randn(4096, 4096)
y = torch.randn(192, 4096, 1)
z = torch.bmm(x.unsqueeze(0).expand_as(192, *x.shape), y)


# C
x = torch.randn(4096, 4096)
y = torch.randn(192, 4096, 1)
z = torch.matmul(y.permute(0, 2, 1), x.t()).permute(0, 2, 1)