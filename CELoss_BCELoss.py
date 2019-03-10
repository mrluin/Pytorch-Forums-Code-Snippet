import torch
import torch.nn as nn
import torch.nn.functional as F

# CrossEntropyLoss
class MyCEModel(nn.Module):
    def __init__(self):
        super(MyCEModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)  # two output neurons = nb_classes


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# BCELoss
class MyBCEModel(nn.Module):
    def __init__(self):
        super(MyBCEModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)  # single output neuron

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    data = torch.randn(5, 10)
    '''
    model = MyCEModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # CELoss target needs no channles dim and dtype is torch.long
    target = torch.randint(0,2,(5,1))# dtype=torch.long)
    print(target.dtype)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    '''

    model = MyBCEModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # target has same shape and dtype as output
    # here needs torch.FloatTensor
    target = torch.randint(0, 2, (5, 1), dtype=torch.float32)

    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()



