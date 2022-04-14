# -*- coding: utf-8 -*-
# https://github.com/ice-tong/pytorch-captcha/blob/master/models.py
# https://www.jianshu.com/p/bd855481eda7
# https://blog.csdn.net/weixin_39903571/article/details/110935320
# https://blog.csdn.net/qq_42951560/article/details/109852790
# https://blog.csdn.net/u013249853/article/details/89393982
# http://guileen.github.io/2019/12/24/understanding-cnn/
import torch
import torch.nn as nn
from models import CNN
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import time

batch_size = 64
base_lr = 0.0001
num_epoches = 100
model_path = './model.pth'

label_num = 2
fin_label = 1

def train():
    transforms = Compose([
        Resize((150, 150)),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = ImageFolder('./data/train', transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, 
                             shuffle=True, drop_last=True)
    #test_data = ImageFolder('./data/train', transform=transforms)
    #test_data_loader = DataLoader(test_data, batch_size=batch_size, 
    #                              num_workers=0, shuffle=True, drop_last=True)
    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epoches):
        start_ = time.time()
        

        cnn.train()
        running_loss = 0.0
        running_acc = 0.0
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * target.size(0) # 累计损失大小，乘积表示当前批次的总损失
            _, pred=torch.max(output,1) # 多分类问题的类别取概率最大的类别
            num_correct = (pred == target).sum() # 当前批次预测正确的个数
            running_acc += num_correct.item() # 累计预测正确的个数

        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
        
        # loss_history = []
        # acc_history = []
        # cnn.eval()
        # for img, target in test_data_loader:
        #     img = Variable(img)
        #     target = Variable(target)
        #     if torch.cuda.is_available():
        #         img = img.cuda()
        #         target = target.cuda()
        #     output = cnn(img)
        #     
        #     acc = calculat_acc(output, target)
        #     acc_history.append(float(acc))
        #     loss_history.append(float(loss))
        # print('test_loss: {:.4}|test_acc: {:.4}'.format(
        #         torch.mean(torch.Tensor(loss_history)),
        #         torch.mean(torch.Tensor(acc_history)),
        #         ))
        
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time()-start_))
        torch.save(cnn.state_dict(), model_path)

if __name__=="__main__":
    train()
    pass