import torch
import torchvision
from torch import nn
from d2l import torch as d2l


batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

input_num,output_num,hiddens=784,10,256

w1=nn.Parameter(torch.randn(input_num,hiddens,requires_grad=True))
b1=nn.Parameter(torch.zeros(hiddens,requires_grad=True))
w2=nn.Parameter(torch.randn(hiddens,output_num,requires_grad=True))
b2=nn.Parameter(torch.zeros(output_num,requires_grad=True))

def relu(x):
    a=torch.zeros_like(x)
    return torch.max(a,x)
params=[w1,b1,w2,b2]
def net(x):
    x=x.reshape((-1,input_num))
    x=relu(x@w1+b1)
    return relu(x@w2+b2)
lr=0.01
trainer=torch.optim.SGD(params,lr)
loss=nn.CrossEntropyLoss()

d2l.train_ch3(net,train_iter,test_iter,loss,10,trainer)


