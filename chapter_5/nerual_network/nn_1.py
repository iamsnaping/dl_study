import torch
from torch import nn
from torch.nn import functional as F



class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer,self).__init__()
    def forward(self,x):
        return x-x.mean()

layer=CenteredLayer()

print(layer(torch.FloatTensor([1,2,3,4,5,6])))

class MyLinear(nn.Module):
    def __init__(self,i_units,units):
        super(MyLinear, self).__init__()
        self.weight=nn.Parameter(torch.rand(i_units,units))
        self.bias=nn.Parameter(torch.rand(units,))
    def forward(self,x):
        h1=torch.matmul(x,self.weight.data)+self.bias.data
        return F.relu(h1)

my_linear=MyLinear(5,3)
print(my_linear.weight.data)

class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.lin1=nn.Linear(20,256)
        self.out=nn.Linear(256,10)
    def forward(self,x):
        return self.out(F.relu(self.lin1(x)))
net=MLP()
x=torch.rand((2,20))
print(net(x))

class MyBlock(nn.Module):
    def __init__(self,*args):
        super(MyBlock, self).__init__()
        for idx,module in enumerate(args):
            self._modules[str(idx)]=module
    def forward(self,x):
        for block in self._modules.values():
            x=block(x)
        return x


# 无法查看结构
class MyBlock2(nn.Module):
    def __init__(self,*args):
        super(MyBlock2, self).__init__()
        self.blocks=[]
        for block in args:
            self.blocks.append(block)
    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x


my_block=MyBlock(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
my_block2=MyBlock2(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(my_block)
print(my_block2)
x=torch.rand((2,20))
print(f'this is my block {my_block(x)}')
print(f'this is my block 2_1{my_block2(x)}')
print(f'this is my block 2_2{my_block2(x)}')

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)
    def forward(self,x):
        x=self.linear(x)
        x=F.relu(torch.mm(x,self.rand_weight)+1)
        x=self.linear(x)
        while x.abs().sum()>1:
            x/=2
        return x.sum()

net=FixedHiddenMLP()
x=torch.rand((2,20))
print(net(x))


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                               nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)
    def forward(self,x):
        return self.linear(self.net(x))

net=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print(net(x))



#--------------------------------------------------------------test---------------------------------------------------------------------------
#-------------------------------------------------@copyright WTC------------------------------------------------------------------------------
#-------------------------------------------------------2022-2800-----------------------------------------------------------------------------
# 1. python list -> can' t print its architecture
#-----------------------------------------------------------2---------------------------------------------------------------------------------

class ParentBlock(nn.Module):
    def __init__(self,net1,net2):
        super(ParentBlock, self).__init__()
        self.net1=net1
        self.net2=net2
        self.linear=nn.Linear(256,10)
    def forward(self,x):
        x1=F.relu(self.net1(x))
        x2=F.relu(self.net2(x))
        x3=torch.cat([x1,x2],dim=0)
        return self.linear(x3)

class SonBlock(nn.Module):
    def __init__(self):
        super(SonBlock, self).__init__()
        self.linear1=nn.Linear(20,256)
        self.linear2=nn.Linear(256,256)
    def forward(self,x):
        x=F.relu(self.linear1(x))
        return self.linear2(x)

class DaughterBlock(nn.Module):
    def __init__(self):
        super(DaughterBlock, self).__init__()
        self.linear1 = nn.Linear(20, 256)
        self.linear2 = nn.Linear(256, 256)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        return self.linear2(x)

p_block=ParentBlock(SonBlock(),DaughterBlock())
print(p_block(x))


class ChildBlock(nn.Module):
    def __init__(self,input_nums,output_nums):
        super(ChildBlock, self).__init__()
        self.lin1=nn.Linear(input_nums,output_nums)
    def forward(self,x):
        return F.relu(self.lin1(x))


class NetBlocks(nn.Module):
    def __init__(self,params):
        super(NetBlocks, self).__init__()
        t=1
        for param in params:
            self._modules[str(t)]=ChildBlock(param[0],param[1])
            t+=1
    def forward(self,x):
        for block in self._modules.values():
            x=block(x)
        return x

params=[[20,128],[128,128],[128,256],[256,128],[128,10]]
net_blocks=NetBlocks(params)
print(f'this is net_blocks :{net_blocks(x)}')
