import torch

epsilon = 1e-8

def masked_softmax(vec, mask, dim=1):

    vec_max = torch.max(vec, dim=1, keepdim=True)[0]

    vec_exp = torch.exp(vec-vec_max) # make vec bigger than zero exp(vec-vec_max)

    vec_exp = vec_exp*(vec == 0).float() # masked vec

    ms = vec_exp / (torch.sum(vec_exp, dim=1, keepdim=True) + epsilon)

    return ms

def masked_softmax1(vec, mask, dim=1):

    masked_vec = vec * mask.float()

    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]

    exps = torch.exp(masked_vec - max_vec)

    masked_exps = exps * mask.float()

    masked_sums = masked_exps.sum(dim, keepdim=True)

    zeros = (masked_sums == 0)

    masked_sums += zeros.float()

    return masked_exps / masked_sums

