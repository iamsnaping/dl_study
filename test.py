

import torch

t1=torch.tensor([1,2,3,4,5,6],dtype=torch.float64).reshape((3,2))
t2=torch.tensor([1,2],dtype=torch.float64)
print(t1,t2)
print(torch.matmul(t1,t2))