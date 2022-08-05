#缓解卷积对位置的敏感性

import torch
from torch import nn
from d2l import torch as d2l

def pool2d(x,pool_size,mode='max'):
    ph,pw=pool_size
    Y=torch.zeros((x.shape[0]-ph+1,x.shape[1]-pw+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=x[i:i+ph,j:j+pw].max()
            elif mode=='avg':
                Y[i,j]=x[i:i+ph,j:j+pw].mean()
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X,(2,2),'avg'))
x=torch.arange(16,dtype=torch.float64).reshape((1,1,4,4))
print(nn.MaxPool2d(3)(x))
print(nn.AvgPool2d(3)(x))
x=torch.cat((x,x+1),1)
print(nn.MaxPool2d(3,padding=1,stride=2)(x))