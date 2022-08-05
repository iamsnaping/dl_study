import torch
from torch import nn
from d2l import torch as d2l

def corr2d_multi_in(x,k):
    return sum([d2l.corr2d(i,j) for i,j in zip(x,k)])
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))


def corr2d_multi_in_out(x,k):
     return torch.stack([corr2d_multi_in(x,i) for i in k],0)
K=torch.stack((K,K+1,K+2),0)
print(corr2d_multi_in_out(X,K))


def corr2d_multi_in_out_1x1(x,k):
    c_out=k.shape[0]
    c_in,h,w=x.shape
    # c_in channels into c line each layer into H*W rows
    x=x.reshape((c_in,h*w))
    k=k.reshape((c_out,c_in))
    return (torch.matmul(k,x)).reshape((c_out,h,w))

X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))

t1=corr2d_multi_in_out_1x1(X,K)
t2=corr2d_multi_in_out(X,K)
print(torch.abs(t1-t2).sum()<1e-6)
