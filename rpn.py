import torch
import torch.nn as nn
import utils
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self,device='cpu'):
        super(RPN, self).__init__()
        self.device=device
        self.conv1=nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            )
        # 判断框内是否包含对象的卷积层
        # 通过此次判断，将不包含对象的框筛除，进入下一阶段，即分类器和建议框
        self.selectLayer=nn.Sequential(
            nn.Conv2d(1024, 18, kernel_size=1),
            nn.ReLU()
        )
        # 先验框和建议框的区别：先验框并不保证框内有待检测对象，而建议框会保证框内有待检测对象
        # 生成先验框的卷积层
        self.boxLayer = nn.Sequential(
            nn.Conv2d(1024, 36, kernel_size=1),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )

        torch.nn.init.xavier_normal_(self.conv1[0].weight.data)
        torch.nn.init.xavier_normal_(self.selectLayer[0].weight.data)
        #torch.nn.init.xavier_normal_(self.boxLayer[0].weight.data)
        torch.nn.init.uniform_(self.boxLayer[0].weight.data,-5,5)

    def forward(self,x,target,mode):# n为batch_size
        n, _, w, _ = x.size()
        x=self.conv1(x)
        score=self.selectLayer(x)#-------------[batch,18,w,h]
        box=self.boxLayer(x)

        # 将box转化为batch*数量*坐标的形式
        box = box.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # box = torch.sigmoid(box)
        box = box * 419
        score=score.permute(0,2,3,1).contiguous().view(n,-1,2)
        score=F.softmax(score,dim=-1)
        iou = []

        # n为batch
        if mode=='train':
            for i in range(n):
                temp = utils.IoU(box[i], target[i])
                iou.append(temp)

            # iou的长度等于batch_size的长度
            batch_logit_loss=torch.tensor(0)
            batch_box_loss = torch.tensor(0)
            batch_logit=[]
            batch_box = []
            # 遍历batch中的所有对象
            j=0
            for i in range(n):
                selected_score,selected_box,selected_iou=self.filter(score[i], box[i],iou[i])
                batch_logit,item_select_loss=self.logit_and_label( selected_iou, selected_score, batch_logit)
                batch_logit_loss = batch_logit_loss + item_select_loss

                batch_box, item_box_loss = self.box_and_label( selected_iou, selected_box, target[i], batch_box)
                if item_box_loss==0:
                    j+=1
                batch_box_loss = batch_box_loss + item_box_loss
                if n-j==0:
                    return batch_logit,batch_box,batch_logit_loss/n,torch.tensor(0)
                else:
                    return batch_logit,batch_box,batch_logit_loss/n,batch_box_loss/(n-j)

        elif mode=='detect':
            return score,box

    def logit_and_label(self,iou,score,batch_logit,iou_threshold=0.7):

        temp, _ = iou.max(dim=0)# temp: 对iou求max

        # 通过iou找出正负例样本
        positive = torch.where(temp >iou_threshold)[0]
        negtive = torch.where(temp < 0.3)[0]
        negtive=negtive[0:len(positive)+1]# 保证正负例样本平衡
                                          # Ensure the balance of positive and negative samples

        # scores=temp[target]
        # 将正负例样本做成one-hot
        label = torch.zeros(1764, 2)
        label = label.to(self.device)
        label[positive, 1] = 1
        label[negtive, 0] = 1

        sample = torch.cat((positive, negtive), dim=0).unique()

        label = label[sample]
        logit = score[sample]
        batch_logit.append(logit)

        item_select_loss = F.cross_entropy(
            logit, label
        )
        del label
        return batch_logit,item_select_loss

    def box_and_label(self,iou,box,target,batch_box,iou_threshold=0.7):
        """
        这个函数用于选出所有iou符合条件的box用于回归训练
        This function is used to select all iou eligible boxes for regression training

        :param iou:
        :param box:
        :param target:
        :param batch_box:
        :param iou_threshold:
        :return:
        """
        # 通过iou筛出所有正样本对应的框的下标
        # 淘汰所有负样本
        # 遍历所有标准框
        temp, box_label = iou.max(dim=0)
        box=box[torch.where(temp>iou_threshold)[0]]# 抛弃一些iou过小的box
        box_label=box_label[torch.where(temp>iou_threshold)[0]]
        temp=temp[torch.where(temp>iou_threshold)]
        if box_label.size()[0]!=0:
            std_box=torch.ones(len(box),4).to(self.device)
            for j in range(target.size()[0]):
                std_box[torch.where(box_label==j)[0],:]=target[j]
            item_box_loss=F.mse_loss(box,std_box)
            batch_box.append(box)
        else:
            item_box_loss=torch.tensor(0).to(self.device)
        #torch.cuda.empty_cache()
        return batch_box,item_box_loss

    def filter(self,score,box,iou):
        """
        该函数用于删除一些不适合进行回归训练的输出
        This function is in order to delete boxes which is too small or too big for regression training

        :param score:
        :param box:
        :param iou:
        :return: (filtered)score,box,iou
        """
        # 保证左上右下结构
        keep = torch.where((box[:, 0] <= box[:, 2]) & (box[:, 1] <= box[:, 3]))[0]
        box = box[keep, :]
        score = score[keep, :]
        iou = iou[:,keep]

        # 去除过小建议框
        keep = torch.where(((box[:, 2] - box[:, 0]) >= 30) & ((box[:, 3] - box[:, 1]) >= 30))[0]#.to(self.device)
        box = box[keep, :]
        score = score[keep, :]
        iou = iou[:,keep]

        # 将建议框压缩至图片大小区间内
        box[:, [0,1,2,3]] = torch.clamp(box[:, [0,1,2,3]], min = 0, max = 419)

        #torch.cuda.empty_cache()
        return score,box,iou