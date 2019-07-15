import torch
import torch.nn as nn


# requires_grad and BatchNorm2d parameters: running_mean, running_var, and num_batches_tracked tracks_running_stats True/False
# BatchNorm2d trainable params are: gamma and beta affine True/False,
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(3, 6, 3),#
                                    nn.BatchNorm2d(6,),
                                    nn.Conv2d(6, 10, 3),
                                    nn.Conv2d(10, 10, 3)])
        self.fcs = nn.Sequential(nn.Linear(320, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 5), #
                                 nn.ReLU(),
                                 nn.Linear(5, 1))

    def forward(self, input):
        out = self.convs[0](input)
        out = self.convs[1](out)
        out = self.convs[2](out)
        out = self.convs[3](out)
        out = out.view(-1, )
        out = self.fcs(out)

        return out

model = Net()
loss = nn.L1Loss()
target = torch.ones(1)

for name, param in model.named_parameters():
    if name == 'convs.0.weight' or name == 'fcs.2.weight':
        # only conv.0.bias and fcs.2.weight = True
        param.requires_grad=True
    else:
        param.requires_grad=False
# output: conv ... fc ... weights and bias

old_state_dict = {}
print(model.state_dict())
#model parameter weight and bias value dict key: value
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad ,model.parameters()), lr=0.001)

for epoch in range(5):

    x = torch.rand(2,3,10,10)
    out = model(x)
    output = loss(out, target)
    output.backward()
    optimizer.step()

new_state_dict = {}
# 看看是否有改变 正常只有convs.0.bias和fcs.2.weight有变化
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

count = 0
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print("Diff in {}".format(key))
        count+=1

print(count)


# Attributes of tensor in computation graph
# following: a is leaf tensor, b and c is non_leaf tensor, do not preserve the intermediate grad, but both requires_grad=True
# a leaf_tensor' requires_grad=False via operations then is also a leaf tensor
# a leaf_tensor' requires_grad=True via operations then is not a leaf tensor
a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a+1
c = b+1
b.retain_grad()
c.backward(torch.ones_like(c))
print(a.grad)
print(a.is_leaf)
print(a.requires_grad)
print(a.retain_grad)
print(b.grad)
print(b.is_leaf)
print(b.requires_grad)
print(b.retain_grad)



# about optimizer.zero_grad()
from torch.autograd.variable import Variable
x = Variable(torch.Tensor([[0]]), requires_grad=True)
for t in range(5):
    y = x.sin()
    y.backward()

print(x.grad) # output x.grad=5 it just do accumulate without grad.zero_()

for t in range(5):
    x.grad.data.zero_()
    y = x.sin()
    y.backward()
print(x.grad) # output x.grad=1
# set grad.zero_() so that the gradients computed previously do not interfere
# with the ones you are currently computing

# about grad_fn
m1 = torch.nn.Conv1d(256, 256, 3, groups=1, bias=False).cuda()
m2 = torch.nn.Conv1d(256, 256, 3, groups=256, bias=False).cuda()
a = torch.randn(1,256,5, device='cuda')
b1 = m1(a)
b2 = m2(a)
print(b1.grad_fn)
print(b2.grad_fn)
print(b1.grad_fn.next_functions)
print(b2.grad_fn.next_functions)
