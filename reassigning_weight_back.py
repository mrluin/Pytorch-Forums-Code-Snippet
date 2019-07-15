import torch
import torch.nn as nn

'''
    # when using unshared tensors, choose .clone() extends requires_grad=True/False
    # when modifying tensors with requires_grad=True, use .data
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1)
        self.conv2 = nn.Conv2d(3,3,3, padding=1)
        self.conv3 = nn.Conv2d(3,3,3, padding=1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

model = Net()
old_params = {}

for name, params in model.named_parameters():
    old_params[name] = params.clone()

# do some modification here
old_params['conv1.weight'][0] = 0.

random_input = torch.randn(1,3,3,3)
random_target = torch.randn(3,3,3)
criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(3):
    opt.zero_grad()
    random_out = model(random_input)
    loss = criterion(random_out, random_target)
    loss.backward()
    opt.step()

# leaf variable with requires_grad=True can not used inplace operation
# Using .data on the other side would work, but is generally not recommended, as changing it after the model was used
# would yield weird results and autograd cannot throw an error.
# So with torch.no_grad():
with torch.no_grad():
    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])

for i in range(3):
    opt.zero_grad()
    random_out = model(random_input)
    loss = criterion(random_out, random_target)
    loss.backward()
    opt.step()





