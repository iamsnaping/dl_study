import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

t=torch.arange(10,dtype=torch.float64).reshape((2,5))
print(t.shape)

t2=torch.arange(1*2*3*4,dtype=torch.float64).reshape((1,2,3,4))
print(t2)
print(t2.mean(dim=(0,2,3),keepdim=True))