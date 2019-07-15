
import torch
import torch.nn as nn


 #Hessian products

linear = nn.Linear(10, 20)
x = torch.randn(1, 10)
y = linear(x).sum()

grad = torch.autograd.grad(y, linear.parameters(), create_graph=True)

v = grad[0].clone().requires_grad_(True)

z = grad[0] @ v.t()

z.mean().backward()
print(linear.weight.grad)
print(z.grad_fn)
print(z.grad_fn.next_functions)
print(z.grad_fn.next_functions[1][0].next_functions)
print(z.grad_fn.next_functions[1][0].next_functions[0][0].next_functions)
'''
<MmBackward object at 0x00000255A34EE9E8>
((None, 0), (<TBackward object at 0x00000255C28464E0>, 0))
((<AccumulateGrad object at 0x00000255C28464E0>, 0),)
'''

