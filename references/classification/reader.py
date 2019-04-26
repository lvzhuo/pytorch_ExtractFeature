from torchvision import transforms, utils
from PIL import Image
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 定义数据的处理方式
data_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize(256),
        # 在256*256的图像上随机裁剪出224*224大小的图像用于训练
        transforms.RandomResizedCrop(224),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir = r'./datasets/Corel5k/images'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')


# 定义自己数据集的数据读入类
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([np.float32(l) for l in information[1:len(information)]])
        self.images = images
        self.labels = np.asarray(labels)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
            # label = self.transform(label)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


def read_data(traindir, valdir, batch_size, num_works):
    # 生成Pytorch所需的DataLoader数据输入格式
    # TrainPath = path + 'train.txt'
    # ValPath = path + "val.txt"

    # dataset, batch_size = args.batch_size,sampler = train_sampler, num_workers = args.workers, pin_memory = True)

    train_Data = my_Data_Set(traindir, transform=data_transforms['train'], loader=Load_Image_Information)
    val_Data = my_Data_Set(valdir, transform=data_transforms['val'], loader=Load_Image_Information)
    train_DataLoader = DataLoader(train_Data, batch_size=batch_size)
    val_DataLoader = DataLoader(val_Data, batch_size=batch_size)


# 验证是否生成DataLoader格式数据

    # for data in train_DataLoader:
    #     inputs, labels = data
    #     print(inputs)
    #     print(labels)
    # for data in val_DataLoader:
    #     inputs, labels = data
    #     print(inputs)
    #     print(labels)

    return train_DataLoader, val_DataLoader


if __name__ == "__main__":

    traindir = r'./datasets/Corel5k/train.txt'
    valdir = r'./datasets/Corel5k/val.txt'
    read_data(traindir, valdir, 1, 1)