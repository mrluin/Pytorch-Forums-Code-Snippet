import torchvision.transforms.functional as TF
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from PIL import Image
import random
import torchvision.transforms as transforms

validation_split = 0.2
shuffle_dataset = True
random_seed = 42
batch_size = 100

data_set_path = ''
train_dataset = glob.glob(os.path.join(data_set_path, '.png'))
dataset_size = len(train_dataset)
indices = list(range(dataset_size))

# split point
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

class MyDataset(Dataset):
    def __init__(self, dataset_path):
        super(MyDataset, self).__init__()

        self.data_list = glob.glob(os.path.join(dataset_path,'.png'))

    def __getitem__(self, index):

        data = Image.open(self.data_list[index])
        # assuming that img_name.png
        file_path = self.data_list[index].split('/')[-1].split('.')[0]
        return data, file_path


    def __len__(self):

        return len(self.data_list)

dataset_path = ''
dataset = MyDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        sampler=train_sampler,)