import torch
import torch.nn as nn


"""
    # calculate mean and std in BN manually
"""


'''# BatchNorm2d test'''
# nte 64 channels in
bn = nn.BatchNorm2d(64)
print(bn)
# for gamma and beta
bn.weight.data.fill_(1)
bn.bias.data.fill_(0)
print(bn.weight)
print(bn.bias.data)

x = torch.rand(6, 64, 224, 224)
# channel <-> batch_size
tmp = x.permute(1, 0, 2, 3).reshape(64, -1)
mu = tmp.mean(dim=1).reshape(1, 64, 1, 1)
sigma = tmp.std(dim=1).reshape(1, 64, 1, 1)
# mu = x.mean(dim=(0,2,3), keepdim=True)
# sigma = x.std(dim=(0,2,3), keepdim=True)         # argument 'dim' must be int, not tuple

# 1e-5 for epsilon
x_ = (x - mu) / (sigma + 1e-5)
# x_ normalized data  approximately equal to bn(x_)
print(x_.shape)
print((bn(x_) - x_).abs().max())
