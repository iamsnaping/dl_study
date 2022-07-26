import torch
from d2l import torch as d2l
from torch import nn
import torchvision

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


def softmax(x):
    x=torch.exp(x)
    partition=x.sum(1,keepdim=True)
    return x/partition
def cross_entrophy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])



def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.gard/batch_size
            param.grad.zeors_()



net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))


def init_weight(x):
    if type(x)==nn.Linear:
        nn.init.normal_(x.weight,std=0.01)
net.apply(init_weight)
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
d2l.train_ch3(net,train_iter,test_iter,loss,10,trainer)

