import torch
import torch.nn as nn

import torch.autograd.anomaly_mode as anomaly_mode


"""
    # Context-manager that enable anomaly detection for the autograd engine.
    # This does two things: - Running the forward pass with detection enabled will allow the backward pass to print the
    # traceback of the forward operation that created the failing backward function. - Any backward computation that generate 
    # “nan” value will raise an error.

    # Note: anomaly detection mode only detect nans that appear during the `.backward()`

    # If you want it for the whole script, add torch.autograd.set_detect_anormly(True)
    # at the beginning. 
"""
'''
    # To check if a tensor contains nans,
    # if tensor.ne(tensor).any():
'''

anomaly_mode.set_detect_anomaly(True)

class MyFunc(torch.autograd.Function):


    @staticmethod
    def forward(ctx, inp):
        return inp.clone()

    @staticmethod
    def backward(ctx, g0):

        raise RuntimeError("Some error in backward")
        return  g0.clone()

def run_fn(a):
    out = MyFunc.apply(a)
    return out.sum()

inp = torch.rand(10,10, requires_grad=True)
out = run_fn(inp)
out.backward()

with torch.autograd.detect_anomaly():
    inp = torch.rand(10,10, requires_grad=True)
    out = run_fn(inp)
    out.backward()


