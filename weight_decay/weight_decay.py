from d2l import torch as d2l
import torch
import torchvision
from torch import nn

n_train,n_test,batch_size,num_inputs=20,100,5,200
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.5
train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
test_data=d2l.synthetic_data(true_w,true_b,n_test)
test_iter=d2l.load_array(test_data,batch_size,is_train=False)

def ini_parameters():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w):
    return torch.sum(w.pow(2))/2

def train(lambd):
    w,b=ini_parameters()
    num_epochs,lr=100,0.003
    net,loss=lambda x:d2l.linreg(x,w,b),d2l.squared_loss
    for epoch in range(num_epochs):
        for X,y in train_iter:
            # with torch.enable_grad():
            l=loss(net(X),y)+l2_penalty(w)*lambd
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size=batch_size)
    print('w的l2范数是',torch.norm(w).item())

train(0.1)