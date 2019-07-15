import torchvision.transforms.functional as TF
#import torchvision.transforms as transforms
import torchvision
import torch


'''
    # does not share memory
    torch.tensor(data, dtype, device, requires_grad) always copies data
    torch.tensor(x) = x.clone().detach()
    torch.tensor(x, requires_grad=True) = x.clone().requires_grad_(True)
    
    torch.Tensor().requires_grad_() or torch.Tensor().detach() to avoid copying

    #both share memory
    torch.from_numpy()
    torch.as_tensor() use Numpy array and avoid copying
'''

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# inplace operation
trans_ = torchvision.transforms.Normalize(mean=mean, std=std)

'''
x = (z-mean) / std
z = x*sigma+mean = (x+mean/sigma)*sigma = (x-(-mean/sigma))/(1/sigma)
'''
#untrans_ = transforms.Normalize(mean = -1*mean/std, std=1/std)

class UnNormalization(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        super(UnNormalization, self).__init__(mean=mean, std=std)

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

