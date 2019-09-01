import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #若能使用cuda，则使用cuda
#LeNet模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
 
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 
 
if __name__ =='__main__':
    # 我们将其转化为tensor数据，并归一化为[-1, 1]。
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                    ])
 
    # 训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
 
    # 将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)
 
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()  # 叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
    for epoch in range(100):  # 遍历数据集100次
        running_loss = 0.0
        # enumerate(sequence, [start=0])，i序号，data是数据
        for i, data in enumerate(trainloader, 0):
 
            inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量],4是batch_size的数值
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)  # 把input数据从tensor转为variable，variable才拥有梯度grad,输入模型训练都要转成Variable
            optimizer.zero_grad()  # 将参数的grad值初始化为0
 
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)  # 将output和labels使用叉熵计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 用SGD更新参数
 
            # 每2000批数据打印一次平均loss值
            running_loss += loss.item()  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0  或使用loss.item()
            if i % 2000 == 1999:  # 每2000批打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
 
    # 测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
    # 将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False)
    correct = 0
    total = 0
    #测试
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(Variable(images))
            value, predicted = torch.max(outputs.data,1)  # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
            # label.size(0) 是一个数
            total += labels.size(0)
            correct += (predicted == labels).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
 
    #输出10分类每个类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()  #4组(batch_size)数据中，输出于label相同的，标记为1，否则为0
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
   
    torch.save(net, 'model.pth')  # 保存模型     
