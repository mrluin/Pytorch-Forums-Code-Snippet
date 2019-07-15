import torch
import torch.nn as nn


'''
Note that slice notation in Python has the syntax

list[<start>:<stop>:<step>]
if step = -1, return a invert version

torch.add() and '+' operation, add each row and column
'''

'''
    # the backpropagation of pow() and sqrt() operation
    # sqrt: better performance, reuse result; pow() doesn't reuse the result
    # reusing in sqrt doesn't allow any inplace operation.
'''


def dummy(x):
    res = torch.mm(x, x.transpose(dim0=0, dim1=1))
    res = res * -2
    res = res + (x**2).sum(dim=1)[:, None]
    res = res + (x**2).sum(dim=1)
    res, idxs = torch.max(res, 0)
    res.flatten()[::x.size(0) + 1] = 0.0
    res = torch.sqrt(res)
    return res


def dummy_pow(x):
    res = torch.mm(x, x.transpose(dim0=0, dim1=1))
    res = res * -2
    res = res + (x**2).sum(dim=1)[:, None]
    res = res + (x**2).sum(dim=1)
    res, idxs = torch.max(res, 0)
    res.flatten()[::x.size(0) + 1] = 0.0
    res = res**(1/2)
    return res

random_input = torch.randn(3,5, requires_grad=True)
random_target = torch.randn(3)
print(dummy(random_input).shape)

criterion = nn.MSELoss()
loss = criterion(dummy_pow(random_input), random_target)
loss.backward()


