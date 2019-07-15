import torch
import torch.nn as nn



'''
    # how to add l regularization to network
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1)


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

y_label = torch.randn(3, 10, 10)
y_pred = torch.randn(3, 10, 10)
batch_size = 10
mdl = Net()
l2_reg = None


for W in mdl.parameters():
    if l2_reg is None:
        l2_reg = W.norm(2)
    else:
        l2_reg = l2_reg + W.norm(2)

batch_loss = (1/batch_size) * (y_pred - y_label).pow(2).sum() + l2_reg*reg_lambda