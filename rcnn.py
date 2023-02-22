import torch
import torch.nn as nn
import torchvision.models as models
from rpn import RPN

class ResNet(nn.Module):
    def __init__(self,pretrain=True):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=pretrain)

        # 定义一个hook函数
        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)# 在第四层中传入该钩子函数，它将自动将第四层的输出钩取出来作为forward函数的输出
                                                            # Pass in the hook function in the fourth layer, it will automatically hook the output of the fourth layer as the output of the forward function

    def forward(self, x):
        self.model(x)
        return self.buffer



class FasterRCNN(nn.Module):
    def __init__(self,device='cpu',mode='train'):
        super(FasterRCNN, self).__init__()
        self.device = device
        self.mode=mode
        self.backbone=ResNet()
        self.rpn=RPN(self.device)

        self.box_num=0

    def forward(self,x,target=None):
        x=self.backbone(x)
        _,_,w,h=x.size()
        self.box_num=w*h*9
        if self.mode=='train':
            cls,box,select_loss,box_loss=self.rpn(x,target,self.mode)
            return cls, box, select_loss, box_loss
        elif self.mode=='detect':
            logit,box=self.rpn(x,target,self.mode)
            return logit,box


