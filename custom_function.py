import torch
import torch.nn as nn
import torch.nn.functional as f

class LinearFunction(torch.autograd.Function):

    # both forward and backward are @staticmethod
    # *  and ** additional arguements
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # save
        ctx.save_for_backward(input, weight, bias)

        # forward calculation
        output = input.mm(weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    # only a single output, gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.


        # needs_input_grad is a tuple of booleans indicating whether
        # each input needs gradient computation

        # Here ctx.need_input_grad[] correspond to the input of forward
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# using a custom function
linear = LinearFunction.apply

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx means context object that can be used to stash info
        ctx.constant = constant
        return tensor*constant

    @staticmethod
    def backward(ctx, grad_output):

        # return two gradients, first for tensor, later for constant value
        return grad_output * ctx.constant , None


# torch.autograd.gradcheck
# 数值近似 evaluated with these tensors are close enough to numerical approximations
from torch.autograd import gradcheck
input = torch.randn(20,20, dtype=torch.double, requires_grad=True)
weight = torch.randn(30,20, dtype=torch.double, requires_grad=True)
test = gradcheck(linear, (input,weight), eps=1e-6, atol=1e-4)
print(test)

# nn exports two kinds of interfaces :
# recommend using modules for all kinds of layers, that hold any parameters or buffers
# recommend using functional form parameter-less operations: activation functions and pooling


'''   for custom module demo  '''
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameters: automatically registered as Modules's parameters once
        # it is assigned as an attribute


        # parameters and buffer need to be registered
        # otherwise they will not appear in .parameters()
        # will not be converted when calling .cuda()

        # nn.Parameters requires gradients by default
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # should always register all possible parameters,
            # but optional ones can be None if you want
            self.register_parameter('bias', None)


        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):

        # using autograd function
        return LinearFunction.apply(input, self.weight, self.bias)


    def extra_repr(self):
        # ***********************
        # (Optional) Set the extra information about this module
        # can be tested by printing an object of this class

        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

