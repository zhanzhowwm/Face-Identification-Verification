# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,20,5,5)#和MNIST不一样的地方，channel要改成3，步长我这里加快了，不然层数太多。
        self.conv2=nn.Conv2d(20,50,4,1)
        self.fc1=nn.Linear(50*6*6,200)
        self.fc2=nn.Linear(200,2)#这个也不一样，因为是2分类问题。
    def forward(self,x):
        #x是一个batch_size的数据
        #x:3*150*150
        x=F.relu(self.conv1(x))
        #20*30*30
        x=F.max_pool2d(x,2,2)
        #20*15*15
        x=F.relu(self.conv2(x))
        #50*12*12
        x=F.max_pool2d(x,2,2)
        #50*6*6
        x=x.view(-1,50*6*6)
        #压扁成了行向量，(1,50*6*6)
        x=F.relu(self.fc1(x))
        #(1,200)
        x=self.fc2(x)
        #(1,2)
        return F.log_softmax(x,dim=1)
