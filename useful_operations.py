import torch
import torch.nn.functional as F
import glob
import os
import numpy as np
import torchvision.transforms.functional as TF
from PIL import  Image

"""
RuntimeError: _thnn_upsample_nearest1d_forward is not implemented for type torch.LongTensor
"""
# Solution
# torch.linspace
N,C,H,W = 1,3,10,10
h,w = 5,5
x = torch.randint(0,10,(N,C,H,W))
iw = torch.linspace(0,W-1, w).long()
ih = torch.linspace(0,H-1,h).long()

x_interp = x[:,:,ih[:,None], iw]
print(x_interp)

# padding to the same dim
target = [[[1, 2, 2], [1, 2, 2, 3, 4]],
          [[8, 9, 10]],
          [[1, 2, 2, 3, 4], [1, 2, 2, 5, 6, 7]]]
# for batch in target for row in batch 双重控制
max_cols = max([len(row) for batch in target for row in batch])
max_rows = max([len(batch) for batch in target])
padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in target]
padded = torch.tensor([row + [0] * (max_cols - len(row)) for batch in padded for row in batch])
padded = padded.view(-1, max_rows, max_cols)
print(padded)

'''# concatenate images and label feed to the network'''

image1 = torch.randint(low=0, high=255, size=(224,224))
image2 = torch.randint(low=0, high=255, size=(224,224))

image1 = Image.fromarray(np.asarray(image1, dtype=np.uint8))
image2 = Image.fromarray(np.asarray(image2, dtype=np.uint8))

image1.save('./folder1/image1.png')
image2.save('./folder2/image2.png',)

f1_folder = './folder1'
f2_folder = './folder2'

f1_images = glob.glob(os.path.join(f1_folder, '*.png'))
f2_images = glob.glob(os.path.join(f2_folder, '*.png'))

for f1_img, f2_img in zip(f1_images, f2_images):

    img1 = Image.open(f1_img)
    img2 = Image.open(f2_img)

    cat_img = torch.cat((TF.to_tensor(img1), TF.to_tensor(img2)), dim=0)# for 'CHW'


'''# for torch.tensor.unfold'''
input = torch.tensor([[[1,2],
                       [3,4]],
                      [[5,6],
                       [7,8]],
                      [[9,10],
                       [11,12]],
                      [[13,14],
                       [15,16]]])
#input.shape = 4, 2, 2
output = torch.tensor([[1,2,5,6],
                       [3,4,7,8],
                       [9,10,13,14],
                       [11,12,15,16]])
output1 = torch.tensor([[1,2],
                        [5,6],
                        [3,4],
                        [7,8],
                        [9,10],
                        [13,14],
                        [11,12],
                        [15,16]])
# shape=4, 4
#input = input.view(1,4,4)
#print(input)
#input_windows = input.unfold(1, 2, 2)
#input_windows = input_windows.unfold(2, 2, 2)
#print(input_windows)
#print(input_windows.contiguous().view(4, 4))
#print(input.shape)
A = input.view(2, 2, 2, 2)
#print(A)
print(A.permute(0,2,1,3))
#print(A.permute(0,2,1,3).contiguous().shape)

x = torch.arange(1, 17).float().view(1, 1, 4, 4)
# x.shape = 1, 1, 4, 4
print(x)
input_windows = x.unfold(2, 2, 2)
# input_windows.shape = 1, 1, 2, 4, 2
#print(input_windows)
input_windows = input_windows.unfold(3 ,2, 2)
# input_windows.shape = 1, 1, 2, 2, 2, 2
#print(input_windows.shape)
input_windows = input_windows.contiguous().view(4,4)
print(input_windows)


atten = torch.randn(1,13) # shape(1,13)
feat = torch.randn(13, 1024, 7, 7)
# want to do sum_i(atten[0][i] * feat[i]) 即atten的13个值分别乘feat的第一个维度
atten = atten.view(13, 1,1,1)
#output = atten[0]* feat
output = atten * feat
#output = torch.einsum('i,ijkl', atten[0], feat)

print(output.shape)
