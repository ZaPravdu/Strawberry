import rcnn
from utils import MyDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast
import tqdm
import os

TRAIN_PATH='./train'
batch_size=8
epochs=8
lr=0.0001

device='cuda' if torch.cuda.is_available() is True else 'cpu'

import torch.nn as nn
import rcnn
import utils
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def train_step(trainloader,model:nn.Module,optimizer,mode):
    # if mode=='total':
    #     for param in model.rpn.boxLayer.parameters():
    #         param.requires_grad = True
    #     for param in model.rpn.selectLayer.parameters():
    #         param.requires_grad = True
    # elif mode == 'cls':
    #     for param in model.rpn.boxLayer.parameters():
    #         param.requires_grad = False
    #     for param in model.rpn.selectLayer.parameters():
    #         param.requires_grad = True
    # elif mode == 'box':
    #     for param in model.rpn.boxLayer.parameters():
    #         param.requires_grad = True
    #     for param in model.rpn.selectLayer.parameters():
    #         param.requires_grad = False
    batch_loss=0
    batch_logit_loss=0
    batch_box_loss=0
    for idx, data in enumerate(trainloader):
        image, box = data
        minus=0
        image=image.to(device)
        for j in range(len(box)):
            box[j]=box[j].to(device)
        logit, reg, logit_loss, box_loss = model(image, box)
        if box_loss==0:
            minus+=1
        batch_logit_loss+=logit_loss
        batch_box_loss+=box_loss
        total_loss=logit_loss+box_loss
        batch_loss+=total_loss
        if idx%16==0:
            print('item loss:',total_loss.item())
            print('box loss:',box_loss.item())
            print('logit loss:',logit_loss.item())
        if mode=='total':
            total_loss.backward()
        elif mode=='cls':
            logit_loss.backward()
        elif mode=='box':
            if box_loss!=0:
                box_loss.backward()
            else:
                logit_loss.backward()# 这个else的目的在于每次计算完loss一定要进行backward，否则计算图不会被释放，最后会爆显存
                                     # The purpose of this else is that every time the loss is calculated, it must be backward,
                                     # otherwise the calculation graph will not be released, and the memory will be exploded in the end.
        optimizer.step()
    #torch.cuda.empty_cache()
    batch_loss /= idx+1
    batch_logit_loss /= idx+1
    batch_box_loss /= idx+1-minus
    return batch_loss,batch_logit_loss,batch_box_loss

if __name__ == '__main__':
    model=rcnn.FasterRCNN(device=device,mode='train')
    model.to(device)

    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    trainset=utils.MyDataset('./train',transform=transform)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=utils.collate_fn,
                             drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    for param in model.backbone.parameters():
        param.requires_grad = False

    for i in range(epochs):
        print('epoch ',i)
        batch_loss,batch_logit_loss,batch_box_loss=train_step(trainloader,model,optimizer,mode='box')
        print('batch_loss:',batch_loss.item())
        print('batch_logit_loss:',batch_logit_loss.item())
        print('batch_box_loss:',batch_box_loss.item())
        # box_loss.backward(retain_graph=True)
        # optimizer.step()
        # print('box loss:', box_loss.item())
        #
        # logit_loss.backward()
        # optimizer.step()
        # print('logit loss:', logit_loss.item())
    print('finish!')


