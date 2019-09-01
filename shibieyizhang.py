import torch
import cv2
import torch.nn.functional as F
from shizhan import Net
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
 
    img = cv2.imread("bird.jpg")  # 读取要预测的图片
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
 
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    output = model(img)
    prob = F.softmax(output,dim=1) #prob是10个分类的概率
    print(prob)
    value, predicted = torch.max(output.data, 1)
   
    pred_class = classes[predicted.item()]
    print(pred_class)
