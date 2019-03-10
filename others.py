import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.util import cropped_dataset
from configs.config import MyConfiguration
from data.dataset import MyDataset
from torch.utils.data import DataLoader
from models import ERFNet
import time




# initial seed for generating random numbers
#print(torch.initial_seed())
# return the random number generator state as torch.ByteTensor
print(torch.get_rng_state())
