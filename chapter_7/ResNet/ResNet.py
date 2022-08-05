from torch import nn
import torch
from d2l import torch as d2l



class ResidualBlock(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1,is_1x1=False):
        super(ResidualBlock, self).__init__()
        self.preprocess=nn.Sequential()
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1)
        self.activate=nn.ReLU()
        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1)
        if is_1x1:
            conv3=nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=stride)
            self.preprocess.add_module('1x1 conv',conv3)
            bn3 = nn.BatchNorm2d(output_channel)
            self.preprocess.add_module('bn', bn3)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.bn2=nn.BatchNorm2d(output_channel)
    def forward(self,x):
        X=self.bn2(self.conv2(self.activate(self.bn1(self.conv1(x)))))
        x=self.preprocess(x)
        return self.activate(X+x)


# x=torch.rand(4,3,6,6)
# blk=ResidualBlock(3,6,is_1x1=True,stride=2)
# Y=blk(x)
# print(Y.shape)
# blk=ResidualBlock(3,3)
# print(blk(x).shape)

def make_block(input_channel,output_channel,num_blocks,first_block=False):
    blocks=[]
    for i in range(num_blocks):
        if i==0 and not first_block:
            blocks.append(ResidualBlock(input_channel,output_channel,is_1x1=True,stride=2))
        else:
            blocks.append(ResidualBlock(output_channel,output_channel))
    return blocks

b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b2=nn.Sequential(*make_block(64,64,2,first_block=True))
b3=nn.Sequential(*make_block(64,128,2))
b4=nn.Sequential(*make_block(128,256,2))
b5=nn.Sequential(*make_block(256,512,2))
net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr,batch_size,num_epochs=0.05,128,20
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())