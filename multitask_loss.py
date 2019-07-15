import torch
import torch.nn as nn
import torch.optim as optim

class custom_lossfunction(nn.Module):
    def __init__(self):
        super(custom_lossfunction, self).__init__()
        self.sigma = nn.Parameter(torch.ones(2))
        self.mse = nn.MSELoss()

    def forward(self, output1, output2, target):

        loss1 = self.mse(output1, target)
        loss2 = self.mse(output2, target)
        #weight = 0.5*self.sigma**2
        loss = 0.5 * loss1/self.sigma[0]**2 + 0.5*loss2/self.sigma[1]**2
        return loss


f1 = nn.Linear(5, 1, bias=False)
f2 = nn.Linear(5, 1, bias=False)

x = torch.randn(3, 5)
y = torch.randn(3)

criterion = custom_lossfunction()
pg = criterion.parameters()
print(criterion.sigma)
opt = torch.optim.SGD(criterion.parameters(), lr=0.1)
loss = criterion(f1(x), f2(x), y)
loss.backward()
opt.step()
print(criterion.sigma)
