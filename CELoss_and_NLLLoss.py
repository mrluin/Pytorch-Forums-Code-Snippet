import torch
import torch.nn as nn
import torch.nn.functional as F


# convert to onehot label
def to_onehot(y, nb_classes):
    # zeros param ->shape
    y_onehot = torch.zeros(y.size(0), nb_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()

    return y_onehot

y = torch.tensor([0, 1, 2, 2])
y_enc = to_onehot(y, 3)
print('one-hot encoding:\n', y_enc)

Z = torch.tensor([[-0.3, -0.5, -0.5],
                  [-0.4, -0.1, -0.5],
                  [-0.3, -0.94, -0.5],
                  [-0.99, -0.88, -0.5]])

def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

# after softmax
smax = softmax(Z)
print(smax)

def to_classlabel(z):
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))


def cross_entropy(softmax, y_target):
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)

xent = cross_entropy(smax, y_enc)
print('Cross Entropy:', xent)

# compared to NLLLoss
print("No reduction compare:")
print("NLLLoss function:",F.nll_loss(torch.log(smax), y, reduction='none'))
print("CELoss function:",F.cross_entropy(Z, y, reduction='none'))

#
print("mean reduction compare:")
print("CELoss function:",F.cross_entropy(Z, y))
print("NLLLoss function:",torch.mean(cross_entropy(smax, y_enc)))