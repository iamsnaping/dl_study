import torch
from d2l import torch as d2l
from torch import nn

def dropout_layer(x,dropout):
    if dropout==0:
        return x
    elif dropout==1:
        return torch.zeros_like(x)
    mask=(torch.rand(x.shape)>dropout).float()
    return mask*x/(1.0-dropout)

drop_out1,drop_out2=0.2,0.5
num_inputs,hidden_layer1,hidden_layer2,num_outputs=784,256,256,10


class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hidden1,num_hidden2
                ,is_training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hidden1)
        self.lin2=nn.Linear(num_hidden1,num_hidden2)
        self.lin3=nn.Linear(num_hidden2,num_outputs)
        self.relu=nn.ReLU()

    def forward(self,x):
        y1=self.relu(self.lin1(x.reshape((-1,self.num_inputs))))
        if self.training==True:
            y1=dropout_layer(y1,drop_out1)
        y2=self.relu(self.lin2(y1))
        if self.training==True:
            y2=dropout_layer(y2,drop_out2)
        y3=self.relu(self.lin3(y2))
        return y3



net=Net(num_inputs, num_outputs,hidden_layer1, hidden_layer2)

batch_size,num_epochs=256,10
lr=0.3

train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
