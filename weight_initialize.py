
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.non_fc = nn.Linear(1, 1)

    def forward(self, x):
        return x


def weight_init(module):
    if isinstance(module, nn.Linear):
        print('initializing layer shape: {}'.format(module.weight.shape))
        nn.init.xavier_normal_(module.weight)


model = MyModel()
[weight_init(m) for name, m in model.named_children() if 'non_fc' not in name]


# nn.Parameters or model.register_params

#Autograd cannot warn you, if you manipulate the underlying tensor.data,
#which might lead to wrong results.
def safe_weight_init(module):
    with torch.no_grad():
        if isinstance(module, nn.Linear):
            print('initializing layer shape: {}'.format(module.weight.shape))
            nn.init.xavier_normal_(module.weight)