from torch import nn
import torch

def conv(X,K):
    h,w=K.shape[0],K.shape[1]
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

X=torch.tensor([0,1,2,3,4,5,6,7,8],dtype=torch.float64).reshape((3,3))
K=torch.tensor([0,1,2,3],dtype=torch.float64).reshape((2,2))
print(conv(X,K))

class Conv2d(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2d, self).__init__()
        self.weight=nn.Parameter(torch.rand(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return conv(x,self.weight)+self.bias
con=Conv2d((2,2))
print(con(X))


conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X = torch.ones((6, 8))
X[:, 2:6] = 0
k=torch.tensor([1,-1],dtype=torch.float64).reshape(1,2)
Y=conv(X,k)
X=X.reshape((1,1,6,8))
Y.reshape((1,1,Y.shape[0],Y.shape[1]))
print(Y)
print(X)
for epoch in range(10):
    l=(Y-conv2d(X))**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:]-=0.03*conv2d.weight.grad
    print(f'epoch {epoch+1} loss {l.sum()}')
print(conv2d.weight.data)
# lr=3e-2
# for i in range(10):
#     Y_hat = conv2d(X)
#     l = (Y_hat - Y) ** 2
#     conv2d.zero_grad()
#     l.sum().backward()
#     # 迭代卷积核
#     conv2d.weight.data[:] -= lr * conv2d.weight.grad
#     if (i + 1) % 2 == 0:
#         print(f'epoch {i+1}, loss {l.sum():.3f}')