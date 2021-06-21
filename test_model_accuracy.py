
import torch
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from models_structures.VGG import * 

MODEL_PATH = 'models/vgg16_cifar10_clean_520_2326.pth'

batch_size = 1
learning_rate = 0.001

IS_MNIST = 0

mean_arr = [0.4914, 0.4822, 0.4465]
stddev_arr = [0.247, 0.243, 0.261]

cifar10_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean_arr,stddev_arr)
])

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x, useFloat=False):
    x = Variable(x)
    return x.cuda(1) if torch.cuda.is_available() else x

if(IS_MNIST):
    # 从torchvision.datasets中加载一些常用数据集
    train_dataset = normal_datasets.MNIST(
    root='./mnist/',  # 数据集保存路径
    train=True,  # 是否作为训练集
    transform=cifar10_transform,  # 数据如何处理, 可以自己自定义
    download=True)  # 路径下没有的话, 可以下载
else:
    # 从torchvision.datasets中加载一些常用数据集
    train_dataset = normal_datasets.CIFAR10(
    root='./datasets/cifar10/',  # 数据集保存路径
    train=True,  # 是否作为训练集
    transform=cifar10_transform,  # 数据如何处理, 可以自己自定义
    download=False)  # 路径下没有的话, 可以下载何处理, 可以自己自定义

# 见数据加载器和batch
if(IS_MNIST):
    test_dataset = normal_datasets.MNIST(root='./mnist/',
                                     train=False,
                                     transform=cifar10_transform)
else:
    test_dataset = normal_datasets.CIFAR10(root='./datasets/cifar10/',
                                     train=False,
                                     transform=cifar10_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

train_loader_for_test = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 两层卷积
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用序列工具快速构建
        self.mode = 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out


device = torch.device("cuda:1")
if (IS_MNIST):
    net = CNN()
else:
    net =VGG('VGG16')

net.load_state_dict(torch.load(MODEL_PATH))
net.to(device)
net.eval()
cleanTotal = 0
cleanRight = 0
for i, (X_test, y_test) in enumerate(test_loader):


    X_test = X_test.cuda(1)
    predict = net(X_test)
    _, predictLabel = torch.max(predict, 1)
    
    if(predictLabel.item() == y_test.item()):
        cleanTotal = cleanTotal + 1
        cleanRight = cleanRight + 1
    else:
        cleanTotal = cleanTotal + 1
    if (i + 1) % 100 == 0:
        print('Iter [%d/%d], accuracy:%d [%d/%d]'
              % (i + 1, len(test_dataset) // batch_size, 10000 * cleanRight / cleanTotal, cleanRight, cleanTotal))
