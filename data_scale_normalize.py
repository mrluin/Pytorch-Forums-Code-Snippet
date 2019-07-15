import torch
import numpy as np
import torchvision.transforms.functional as TF

mean = [0.5, 0.5, 0.5]
var = [0.5, 0.5, 0.5]
'''
map = Image.open('2.tif')
print("original map")
map_array = np.asarray(map)
print(map_array.shape)
print(map_array.dtype)
'''
map = torch.randint(low=0, high=255, size=(3, 3, 3), dtype=torch.uint8)
map_array = np.asarray(map)
print(map_array)
to_tensor_map = TF.to_tensor(map_array)
print("scaled map")
print(to_tensor_map.shape)
print(to_tensor_map)
normal_map = TF.normalize(to_tensor_map, mean, var)
print("normal_map")
print(normal_map)

# manaul_norm
mean = torch.mean(to_tensor_map, dim=(0, 1))
var = torch.var(to_tensor_map, dim=2, keepdim=False, unbiased=False)

manaul_norm_map = (to_tensor_map - mean) / var
print(manaul_norm_map)