import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1=nn.Linear(20,256)
        self.lin2=nn.Linear(256,10)
    def forward(self,x):
        return self.lin2(F.relu((self.lin1(x))))


net=MLP()
x=torch.rand((2,20))
print(net(x))
torch.save(net.state_dict(),'mlp.params')
net1=MLP()
net1.load_state_dict(torch.load('mlp.params'))
net1.eval()
print(net1(x))
