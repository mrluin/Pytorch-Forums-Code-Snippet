import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


def bench(module, n=1000):
    forward_total = 0
    backward_total = 0
    for _ in range(n):
        x, y = next(iter(dataloader))
        x = x.view(-1, 784)
        now = time.perf_counter()
        out = module(x)
        forward_total += time.perf_counter() - now
        loss = loss_fn(out, y)
        now = time.perf_counter()
        loss.backward()
        backward_total += time.perf_counter() - now
    return forward_total / n, backward_total / n


class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        #ctx.x = x
        #ctx.weight = weight
        #ctx.bias = bias
        ctx.save_for_backward(x, weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        return grad_out @ weight, grad_out.t() @ x, grad_out.sum(dim=0)


class MyLinear(nn.Linear):
    def forward(self, x):
        return MyLinearFunction.apply(x, self.weight, self.bias)


mnist_path = 'mnist_data'
mnist = MNIST(
    mnist_path,
    train=True,
    transform=Compose([
        ToTensor(),
        lambda x: x[0]  # mnist images contain one channel
    ]),
    download=True
)
dataloader = DataLoader(mnist, batch_size=128)
loss_fn = nn.CrossEntropyLoss()

m1 = nn.Linear(784, 10)
m2 = MyLinear(784, 10)
m1.weight.data = m2.weight.data.clone()
m1.bias.data = m2.bias.data.clone()

vf, vb = bench(m1)
cf, cb = bench(m2)
print('vanilla forward: {:.6f}, vanilla backward: {:.6f}'
      .format(vf, vb))
print('custom  forward: {:.6f}, custom  backward: {:.6f}'
      .format(cf, cb))


'''
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = MyLinear()
model.fc2.register_forward_hook(get_activation('fc2'))
'''
