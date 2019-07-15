import torch
import torch.utils.data as data
import torch.nn as nn


torch.manual_seed(1)
ignore_class = 5
label = torch.randint(low=0, high=5, size=(3,3)) # 0-4
label[label==4] = 5

label_one_hot = torch.zeros(6,3,3).scatter_(0, label.unsqueeze(0), 1)

label_one_hot_ignore = label_one_hot[:3]
print(label_one_hot_ignore.shape)