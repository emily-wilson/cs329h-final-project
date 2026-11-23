import torch
import math
import torch.nn.functional as F

def reparameterization(mean, std):
    epsilon = torch.randn_like(std).to(mean.device)  # sampling epsilon
    z = mean + std * epsilon                         # reparameterization trick
    z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
    return z