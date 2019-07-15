import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, ConcatDataset, Subset
from torch.utils.data import SequentialSampler, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import BatchSampler, RandomSampler


'''
# data.random_split
# data.Dataset
# data.Sampler
'''

# TensorDataset
data1 = torch.randn(10,3,3,3)
dataset = TensorDataset(data1)
print(dataset)

# ConcatDataset 可以concat不相同尺寸的img
data2 = torch.randn(10, 5,5,5)
dataset2 = TensorDataset(data2)
concat_dataset = ConcatDataset([dataset, dataset2])
print(concat_dataset)

# Subset pass a dataset and indices 如果indice out of range,不会报错也不会输出
subset_dataset = Subset(concat_dataset, [0,9,10,19])


# data.random_split pass a dataset and splits' length: sequence
print(len(concat_dataset))
data.random_split(concat_dataset, [5,5,5,5])

# Sequential_Sampler for what ???
sampler1 = SequentialSampler(concat_dataset)
dataloader = DataLoader(dataset = dataset, sampler=sampler1)
#print(next(iter(dataloader)))

''' 
for replacement :L if or not samples from the same row.
当num_samples的值大于len(dataset)的时候, with replacement=True可以重复取
'''
# RandomSampler: num_samples -> how many samples to sample with replacement=True,
sampler2 = RandomSampler(data_source=concat_dataset)
dataloader2 = DataLoader(dataset= concat_dataset, sampler=sampler2)

# SubsetRandomSampler : only random the indices given
sampler3 = SubsetRandomSampler([0, 9, 10, 19])

# WeightedRandomSampler :
torch.manual_seed(1)
weights = [1, 100]  # not must sum to 1
data3 = torch.randint(0,5,size=(2,))
dataset3 = TensorDataset(data3)
sampler4 = WeightedRandomSampler(weights, num_samples=3, replacement=True)
dataloader4 = DataLoader(dataset3, sampler=sampler4)

# BatchSampler  v.s. dataloader
list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))

# distributed sampler ... ... 