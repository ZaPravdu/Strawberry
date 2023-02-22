import torch
from torch.utils.data import DataLoader,Dataset
from PIL import Image,ImageDraw
import os
import json
from torchvision import transforms
import skimage


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.labels = os.listdir(self.root_dir)
        for i in self.labels:
            if i[-4:] == '.jpg':
                self.labels.remove(i)
        self.transform = transform
        # getitem内置方法用于获取数据中的图片与标签

    def __getitem__(self, index):

        label_path = os.path.join(self.root_dir, self.labels[index])
        js = json.load(open(label_path))
        image_path = os.path.join(self.root_dir, js['imagePath'])
        shape = get_box(js)
        img = Image.open(image_path)
        # label=torch.ones(3,1)
        # 这段代码用于将图片进行预处理
        if self.transform is not None:
            img = self.transform(img)

        return img, shape

    def get_label(self, index):
        label_path = os.path.join(self.root_dir, self.labels[index])
        js = json.load(open(label_path))
        shape = get_box(js)
        return shape

    def __len__(self):
        return len(self.labels)

def coordinate2box(cdn):
    x1=cdn[0]
    y1=cdn[1]
    x2=cdn[2]
    y2=cdn[3]
    return [x1,y1,x2,y2]

def get_label(js:dict):
    mask=[]
    for i in range(len(js['shapes'])):
        X = []
        Y = []
        for j in js['shapes'][i]['points']:
            Y.append(j[0])
            X.append(j[1])
            mask.append([X,Y])
    shape=[]
    for i in mask:
        # 此时i是mask的集合中的一个，i中含有两个集合，x坐标集和y坐标集
        temp=[]# 空列表temp同于存放shape的各项参数
        x_min = i[0].min()
        temp.append(x_min)
        y_min = i[1].min()
        temp.append(y_min)
        x_max=i[0].max()# 对x坐标集求最大值，得到这个mask的横坐标的最大值
        temp.append(x_max)
        y_max = i[1].max()
        temp.append(y_max)
        center=[(x_min+x_max)/2,(y_min+y_max)/2]
        temp.append(center)
        shape.append(temp)

    # 此时shape包含若干个mask的shape，每个元素内部存放：x_min,x_max,y_min,y_max,center
    return mask,shape


def get_box(js: dict):
    edge = []
    for i in range(len(js['shapes'])):
        X = []
        Y = []
        for j in js['shapes'][i]['points']:
            Y.append(j[0])
            X.append(j[1])
        edge.append([X, Y])
    # edge: [[X1,Y1],[X2,Y2],...]

    raw_mask = []
    for i in edge:
        rr, cc = skimage.draw.polygon(i[0], i[1])
        raw_mask.append([cc, rr])
    # mask: [[colum1,row1],[colum2,row2],...]

    shape = []
    for i in edge:
        # 此时i是mask的集合中的一个，i中含有两个集合，x坐标集和y坐标集
        temp = []  # 空列表temp同于存放shape的各项参数
        x_min = min(i[0])
        temp.append(x_min)

        y_min = min(i[1])
        temp.append(y_min)

        x_max = max(i[0])  # 对x坐标集求最大值，得到这个mask的横坐标的最大值
        temp.append(x_max)

        y_max = max(i[1])
        temp.append(y_max)
        #         center=[(x_min+x_max)/2,(y_min+y_max)/2]
        #         temp.append(center)
        shape.append(temp)
        for i in range(len(shape)):
            shape[i] = torch.Tensor(shape[i])

    shape = torch.stack(shape)
    # 此时shape包含若干个mask的shape，每个元素内部存放：x_min,x_max,y_min,y_max,center
    # shape: [[x_min1,x_max1,y_min1,y_max1,center],...]

    return shape

# box1是输入，box2是标签
def IoU(box1, box2, batch_size=16):  # 输入为[batch＊框数＊４]
    """
    只能计算一张图片输出的回归框对于所有标准框的iou
    """
    IoUs = []
    # 解析出标准框的数目
    n, _ = box2.size()

    # 对不符合左上右下格式的张量进行转换
    # 检查x1是否大于x2

    target = torch.where(box1[:, 0] > box1[:, 2])[0]
    temp = box1[target, :]
    box1[target, 0] = temp[:, 2]
    box1[target, 2] = temp[:, 0]
    # 检查y1是否大于y2

    target = torch.where(box1[:, 1] > box1[:, 3])[0]
    temp = box1[target, :]
    box1[target, 1] = temp[:, 3]
    box1[target, 3] = temp[:, 1]

    #     if box2[:,0] > box2[:,2]:
    #         temp = box2[0]
    #         box2[0] = box2[2]
    #         box2[2] = temp

    #     if box2[1] > box2[3]:
    #         temp = box2[1]
    #         box2[1] = box2[3]
    #         box2[3] = temp

    # 计算box的面积
    for i in range(n):
        s_box1 = torch.abs((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))
        s_box2 = torch.abs((box2[i][2] - box2[i][0]) * (box2[i][3] - box2[i][1]))

        x1 = torch.max(box1[:, 0], box2[i][0])
        y1 = torch.max(box1[:, 1], box2[i][1])

        x2 = torch.min(box1[:, 2], box2[i][2])
        y2 = torch.min(box1[:, 3], box2[i][3])
        zero = torch.tensor(0)

        w, h = torch.max(zero, x2 - x1), torch.max(zero, y2 - y1)
        s_intersec = w * h
        # s_intersec=s_intersec.unsqueeze(0).unsqueeze(2)
        iou = s_intersec / (s_box1 + s_box2 - s_intersec)
        IoUs.append(iou)
    IoUs = torch.stack(IoUs)
    return IoUs


def collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.stack(images)
    return images, bboxes