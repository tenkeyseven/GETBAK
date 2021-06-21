from PIL.Image import LANCZOS
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from livelossplot import PlotLosses
from backdoor_utils.train_model import train_model
import backdoor_utils.malicious_data_loader as malicious_data_loader
from models_structures.VGG import *
from rich.progress import track

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x, useFloat=False):
    x = Variable(x)
    return x.cuda(1) if torch.cuda.is_available() else x

# TODO configuation 配置文件，后续更新将使用一个统一的配置文件
# --------------------------------------------------------
MODEL_TYPE = 'VGG16'
DATASETS_TYPE = 'CIFAR10'
# -------------------------------------------------------
# FOOLMODEL could be 'VGG16-CIFAR10', 'RESNET18_IMAGENETTE'
FOOLMODEL = 'VGG16_CIFAR10' 
# --------------------------------------------------------
# 攻击者指定的靶向目标
ATTACK_TARGET = 7
# 干净模型调用地址，以下是干净模型地址
CLEAN_MODEL_PATH_VGG16_CIFAR10 = 'models/vgg16_cifar10_clean_520_2326.pth'
CLEAN_MODEL_PATH_RESNET18_IMAGENETTE = 'models/resnet18_imagenette_clean.pth'
# 对模型进行选择
CLEAN_MODEL_PATH = CLEAN_MODEL_PATH_VGG16_CIFAR10
# 后门模型保存地址
BACKDOOR_MODEL_PATH = 'models/vgg16_cifar10_backdoor_520_2326.pth'
# 投毒比率
BACKDOOR_RATE = 0.5
# 投毒数据的 Batch Size
BATCH_SIZE = 100
# 使用基本训练方式的参数
num_epochs = 15
learning_rate = 0.001
# 触发器位置
TRIGGER_PATH = 'share_data/trigger/trigger_cifar10_520.png'
# --------------------------------------------------------

if FOOLMODEL == 'VGG16_CIFAR10':
    input_size = 32
    unl_mean = [-0.4914/0.247, -0.4822/0.243, -0.4465/0.261]
    unl_std = [1/0.247, 1/0.243, 1/0.261]
elif FOOLMODEL == 'RESNET18_IMAGENETTE':
    input_size = 224
    unl_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    unl_std = [1/0.229, 1/0.224, 1/0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]),
}

transform_unormalize=transforms.Compose([
        transforms.Normalize(unl_mean,unl_std),
        transforms.ToPILImage()

])

# --------------------------------------------------------
# 读取原始数据段
# --------------------------------------------------------
if FOOLMODEL == 'RESNET18_IMAGENETTE':
    # 对从文件夹中读取训练图片
    # 读取原始数据集，PIL文件格式，进行剪裁，ToTensor处理
    data_dir = './datasets/imagenette/imagenette2'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}
elif FOOLMODEL == 'VGG16_CIFAR10':
    # 用 torchvison 方法来获取数据级
    # 先创建一个空字典
    image_datasets = {x:[] for x in ['train', 'val']}
    image_datasets['train'] = torchvision.datasets.CIFAR10(
        root='./datasets/cifar10',
        train=True,
        transform=data_transforms['train'],
        download=False
    )
    image_datasets['val'] = torchvision.datasets.CIFAR10(
        root='./datasets/cifar10',
        train=False,
        transform=data_transforms['val'],
        download=False
    )

# --------------------------------------------------------
# 对训练数据进行恶意处理，通过调用 CreateMaliciousDataset 来实现
# --------------------------------------------------------

# 对训练数据集，通过恶意数据 loader 进行数据处理

# 恶意数据集生成，Tensor->PIL->Tensor (经过Normalize)
# dataset_mal = malicious_data_loader.CreateMaliciousDataset(image_datasets['train'], poison_target=7, dataMode='mal_mix')

print('loading malicious data')
dataset_mal = malicious_data_loader.CreateMaliciousDataset(image_datasets['train'], poison_target=ATTACK_TARGET, dataMode='mal_mix', portion = BACKDOOR_RATE)

# 先创建一个空dataloaders
dataloaders = {x:[] for x in ['train','val']}

# 对恶意更改后的数据集合进行 loader 载入
dataloaders['train'] = torch.utils.data.DataLoader(dataset_mal, batch_size=BATCH_SIZE, shuffle=True, num_workers=64)

# 对测试数据集，直接将测试集数据载入dataloader中
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=64)
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=64)
#               for x in ['val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 用于测试的函数
def show_img():
    """fuction which unnormalize the tensor picture and transform it to PIL Image.
    """  
    print('dataset_sizes =', dataset_sizes)

    for batch, (img, label) in enumerate(dataloaders['train']):
        if batch <1:
            continue
        for i in range(len(img)):
            if label[i] == 7:
                iimg = transform_unormalize(img[i])
                iimg.save('crafted_CIFAR10/7/test{}.png'.format(i))
            else:
                iimg = transform_unormalize(img[i])
                iimg.save('crafted_CIFAR10/test{}.png'.format(i))
        break

# show_img()

if FOOLMODEL == 'RESNET18_IMAGENETTE':
    model_ft = models.resnet18()
    # Finetune Final few layers to adjust for tiny imagenet input
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
    # Multi GPU
    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #Train
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

    torch.save(model_ft.state_dict(), BACKDOOR_MODEL_PATH)

elif FOOLMODEL == 'VGG16_CIFAR10':
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
        for i, (images, labels) in enumerate(dataloaders['train']):
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
                        len(dataset_mal) // BATCH_SIZE, loss.item()))

        print('[%d,%d] loss:%.03f' % (epoch + 1, num_epochs, sum_loss / len(dataloaders['train'])))
        print('        correct:%.03f%%' % (100 * train_correct / len(dataset_mal)))
    torch.save(cnn.state_dict(), BACKDOOR_MODEL_PATH)