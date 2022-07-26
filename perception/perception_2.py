import torch
from d2l import torch as d2l
from torch import nn

net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
batch_size=256
def init_weight(x):
    if type(x)==nn.Linear:
        nn.init.normal_(x.weight,std=0.01)
net.apply(init_weight)
loss=nn.CrossEntropyLoss(reduction='none')
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net,train_iter,test_iter,loss,10,trainer)
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def predict_(net,test_iter,n=6):
    for X,y in test_iter:
        break
    true=get_fashion_mnist_labels(y)
    pred=get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(true, pred)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()
predict_(net,test_iter)