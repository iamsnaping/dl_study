from d2l import torch as d2l
import torch
from torch import nn
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,10))
trainer=torch.optim.SGD(net.parameters(),lr=0.3)
loss=nn.CrossEntropyLoss(reduction='none')
num_epochs,batch_size=10,256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

def init_weigt(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weigt)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)