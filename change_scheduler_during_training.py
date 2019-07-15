import torch
import torch.nn as nn

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
random_input = torch.randn(1,3,3,3)
random_target = torch.randn(3,3,3)

opt = torch.optim.SGD(model.parameters(), lr=2e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)

criterion = nn.MSELoss()

print(lr_scheduler.state_dict())
for step in range(1, 11):
    if step == 6:
        #opt.param_groups[0]['lr'] = 1e-4
        lr_scheduler.base_lrs=[1e-4]
        #lr_scheduler.last_epoch=-1
        print('change', lr_scheduler.state_dict())

    if step < 6:
        lr_scheduler.step(step)
    else:
        lr_scheduler.step(step-5)
    print('lr_scheduler: ',lr_scheduler.get_lr())
    print('optimizer_lr: ',opt.param_groups[0]['lr'])
    opt.zero_grad()
    output = model(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    opt.step()
