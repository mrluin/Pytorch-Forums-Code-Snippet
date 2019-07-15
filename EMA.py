import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(3,3,3, padding=1, bias=False)
        )
    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

model = Net()

class EMA():
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.momentum = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, input):
        assert name in self.shadow.keys()
        new_average = self.momentum * self.shadow[name] + (1-self.momentum) * input
        self.shadow[name] = new_average.clone()
        return new_average

ema = EMA(0.999)
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        ema.register(name, parameter.data)

# training
# the EMA using in training phase
# for batch_data in dataloader:
    optimizer.zero_grad()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = ema.shadow[name]

