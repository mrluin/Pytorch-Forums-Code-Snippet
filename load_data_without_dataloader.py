import numpy as np
import torch
from torch.utils.data import TensorDataset
import random
import more_itertools


def load_data():
    # Fake data. You can also load your images and convert them into tensors.
    number_images = 100
    images = torch.randn(number_images, 3, 2, 2)
    labels = torch.ones(number_images, 1)
    return TensorDataset(images, labels)


def get_batch(dataset, batch_idx):
    ''' Returns the data items given batch indexes '''

    # Set up the datastructures
    im_size = dataset[0][0].size()
    batch_size = len(batch_idx)
    batch_data = torch.empty((batch_size, *im_size))
    batch_labels = torch.empty((batch_size, 1))

    # Add data to datastructures
    for i, data_idx in enumerate(batch_idx):
        data, label = dataset[data_idx]
        batch_data[i] = data
        batch_labels[i] = label

    return batch_data, batch_labels


dataset = load_data()
data_length = len(dataset)

batch_size = 10
n_epochs = 10
for epoch in range(n_epochs):
    # Create indexes, shuffles them and split them into batches
    indexes = list(range(data_length))
    random.shuffle(indexes)
    indexes = more_itertools.chunked(indexes, batch_size)

    for batch_idx in indexes:
        images, labels = get_batch(dataset, batch_idx)
        # You can now work with your data