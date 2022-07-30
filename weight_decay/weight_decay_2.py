import torch
from d2l import torch as d2l
from torch import nn


n_train,n_test,batch_size,num_inputs=20,100,5,200
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.5
train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
test_data=d2l.synthetic_data(true_w,true_b,n_test)
test_iter=d2l.load_array(test_data,batch_size,is_train=False)

def train(wd):
    loss=nn.MSELoss()
    net=nn.Sequential(nn.Linear(num_inputs,1))
    num_epochs,lr=100,0.003
    for params in net.parameters():
        params.data.normal_()
    trainer=torch.optim.SGD([{'params':net[0].weight,'weight_decay':wd},{'params':net[0].bias}],lr=lr)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            trainer.step()
    print(net[0].weight.norm().item())
train(3)