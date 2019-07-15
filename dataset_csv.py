from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import torch
import h5py
# https://discuss.pytorch.org/t/dataloader-on-two-datasets/18504/2

'''
	                ClassLabel
000001.jpg	0
000002.jpg	0
000003.jpg	1

'''

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, txt_path, img_dir, transform=None):
        df = pd.read_csv(txt_path, sep=" ", index_col=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
        self.y = df['ClassLabel'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
# for train_set1
train_dataset1 = CelebaDataset(txt_path='celeba_gender_attr_train.txt',
                              img_dir='img_align_celeba/',
                              transform=custom_transform)
# for train_set2
#train_dataset2

train_loader_1 = DataLoader(dataset=train_dataset1,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

# for test_set1
test_dataset1 = CelebaDataset(txt_path='celeba_gender_attr_test.txt',
                              img_dir='img_align_celeba/',
                              transform=custom_transform)

# for test_set2
#test_dataset2
test_loader_1 = DataLoader(dataset=test_dataset1,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

#
num_epochs= 20


for epoch in range(num_epochs):
    for batch_idx, features in enumerate(train_loader_1):
        # train model on the training dataset #1
        pass
    for batch_idx, features in enumerate(train_loader_2):
        # train model on the training dataset #2
        pass

for batch_idx, features in enumerate(test_loader1):
    # evaluate model on test dataset #1
    pass
for batch_idx, features in enumerate(test_loader2):
    # evaluate model on test dataset #2
    pass


'''dataset from h5py'''

class dataset_h5(torch.utils.data.Dataset):
    """
    Reads in a dataset
    """
    def __init__(self, in_file, mode = 'training'):
        super(dataset_h5, self).__init__()
        self.file_path = in_file
        self.dataset_mode = mode
        self.dataset = None

        with h5py.File(self.file_path, 'r') as file:
            self.n_images, self.channels,self.nx, self.ny = file[self.dataset_mode]['inputs'].shape
            self.n_images_check, self.nfeatures = file[self.dataset_mode]['labels'].shape

        if self.n_images != self.n_images_check:
            print('Number of input samples does not match number of label samples!')

        norm_set = h5py.File(in_file, 'r')['normalization']
        self.data_mean = norm_set['mean'][0]
        self.data_std = norm_set['std'][0]
        self.transform =transforms.Normalize((self.data_mean,), (self.data_std,))

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.dataset_mode]
        input = self.transform(torch.tensor(self.dataset['inputs'][index, :, :].astype('float32')))
        labels = torch.tensor(self.dataset['labels'][index, :].astype('float32'))

        return input, labels

