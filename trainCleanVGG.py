from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from models_structures.VGG import *
from models_structures.SimpleNN import *
from rich.progress import track

num_epochs = 30
batch_size = 100
learning_rate = 0.001
IS_MNIST = 0

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x, useFloat=False):
    x = Variable(x)
    return x.cuda(1) if torch.cuda.is_available() else x

mean_arr = [0.4914, 0.4822, 0.4465]
stddev_arr = [0.247, 0.243, 0.261]

cifar10_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean_arr,stddev_arr)
])

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
    download=False)  # 路径下没有的话, 可以下载

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


if (IS_MNIST):
    cnn = SimpleNN()
else:
    cnn =VGG('VGG16')

if torch.cuda.is_available():
    cnn = cnn.cuda(1)

cnn.train()
print('Train Starts')
# 选择损失函数和优化方法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
for epoch in track(range(num_epochs)):
    sum_loss = 0.0
    train_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        
        outputs = cnn(images)
        optimizer.zero_grad()
        loss = loss_func(outputs, labels)
        y = loss.backward()
        optimizer.step()
        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        train_correct += torch.sum(id == labels.data)
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                      len(train_dataset) // batch_size, loss.item()))

    print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss / len(train_loader)))
    print('        correct:%.03f%%' % (100 * train_correct / len(train_dataset)))


torch.save(cnn.state_dict(), './models/vgg16_cifar10_clean_xx.pth')
