import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear1 = nn.Linear(100,50)
        self.linear2 = nn.Linear(50,40)
        self.linear3 = nn.Linear(40,40)

    def forward(self, input):
        out1 = self.linear1(input)
        out2 = self.linear2(out1)
        out3 = self.linear3(out2)

        return out2, out3


if __name__ == '__main__':

    # model and submodel
    x = torch.randn((1,100), requires_grad=True)
    y = torch.randn(1,40)
    criterion = nn.MSELoss()
    model = ToyModel()
    out2, out3 = model(x)
    out2.retain_grad()
    out3.retain_grad()
    #print(out2.requires_grad, out3.requires_grad)
    #print(out3)
    loss1 = criterion(out2, y)
    loss2 = criterion(out3, y)

    loss = loss1+loss2
    loss.backward()

    '''
    # in this mode, need retain_graph
    loss1.backward(retain_graph=True)
    loss2.backward()
    '''
    '''
    # output non_leaf variable grad need retain_grad()
    # otherwise only save is_leaf variable grad
    print(out2.grad)
    print(out3.grad)
    '''

