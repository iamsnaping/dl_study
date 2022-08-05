import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_col,in_channel,out_channel):
  layer=[]
  for i in range(num_col):
    layer.append(nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=1))
    layer.append(nn.ReLU())
    in_channel=out_channel
  layer.append(nn.MaxPool2d(kernel_size=2,stride=2))
  return nn.Sequential(*layer)

v_archive=((1,64),(1,128),(2,256),(2,512),(2,512),(2,512))

def vgg(v_archive):
  in_channel=1
  layer=[]
  for (num_cols,out_channel) in v_archive:
    layer.append(vgg_block(num_cols,in_channel,out_channel))
    in_channel=out_channel
  return nn.Sequential(*layer,nn.Flatten(),nn.Linear(out_channel*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
  nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
  nn.Linear(4096,10)
  )
net=vgg(v_archive)
batch_size=128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
lr,num_epochs=0.05,10
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
