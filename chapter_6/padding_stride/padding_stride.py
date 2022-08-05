import torch
from torch import nn
# n-k+p+1
con2d1=nn.Conv2d(1,1,kernel_size=3,padding=1)
X=torch.ones((8,8)).reshape((1,1,8,8))
Y=con2d1(X)
print(Y.shape)

#(n-k+p+sh)/sh 224-11+1+4/4

con2d2=nn.Conv2d(1,1,kernel_size=5,padding=1,stride=2)
Y=con2d2(X)
print(Y.shape)

con2d3=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=2)
Y=con2d3(X)
print(Y.shape)


