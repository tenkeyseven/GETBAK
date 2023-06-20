# i = 6
# plt.title('Model Predict:{}  Ground Truth:{}'.format(label_classes[predicted.cpu()[i]],label_classes[labels[i]]))
# plt.imshow(np.transpose(transforms_unnormalize(images[i].cpu()),(1,2,0)))

# plt.figure(figsize=[8,6],dpi=144)
# plt.xticks(rotation=30)
# plt.bar(label_classes, output_labels_distribution, facecolor = 'r')
# for x,y in zip(label_classes, output_labels_distribution):
#     plt.text(x,y,'%d'%y, ha = 'center', va='bottom')
# plt.bar(label_classes, output_labels_distribution_2, facecolor = 'g')
# for x,y in zip(label_classes, output_labels_distribution_2):
#     plt.text(x,y,'%d'%y, ha = 'center', va='bottom')
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from livelossplot import PlotLosses
from numpy.lib.type_check import real
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
from rich.progress import track
from models_structures.VGG import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
transforms_2tensor = transforms.ToTensor()
transforms_2pil = transforms.ToPILImage()

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
BACKDOOR_RATE = 0.0
# 投毒数据的 Batch Size
BATCH_SIZE = 100
# 使用基本训练方式的参数
num_epochs = 15
learning_rate = 0.001
# 触发器位置
TRIGGER_PATH = 'share_data/trigger/trigger_cifar10_520.png'
# --------------------------------------------------------

transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transforms_unnormalize = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

if FOOLMODEL == 'VGG16_CIFAR10':
    resize =  32
    input_size = 32
    unl_mean = [-0.4914/0.247, -0.4822/0.243, -0.4465/0.261]
    unl_std = [1/0.247, 1/0.243, 1/0.261]
    mean_arr = [0.4914, 0.4822, 0.4465]
    stddev_arr = [0.247, 0.243, 0.261]
    label_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

elif FOOLMODEL == 'RESNET18_IMAGENETTE':
    resize = 256
    input_size = 224
    unl_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    unl_std = [1/0.229, 1/0.224, 1/0.225]
    mean_arr = [0.485, 0.456, 0.406]
    stddev_arr = [0.229, 0.224, 0.225]
    label_classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
    ]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]),
}

transforms_normalize = transforms.Normalize(mean_arr, stddev_arr)
transforms_unnormalize = transforms.Normalize(unl_mean, unl_std)


#%%
def show_some_images(model_type, imgs, output_labels, real_labels, length):
    """
    print labels
    """
    plt.figure(figsize=(8, 6))
    for i in range(length):
        plt.subplot(1,length, i+1)
        plt.axis('off')
        plt.title('{} Model Predict: {}, Groundtruth: {}'.format(model_type, output_labels[i], real_labels[i]))
        plt.imshow(imgs[i])

# %%
# Matplotlib - Bar
def plot_bar(distribution1, distribution2):
    """
    plot double bar
    """
    plt.figure(figsize=[12,8],dpi=144)
    plt.xticks(rotation=30)
    plt.ylabel('Amount of Model Prediction')
    plt.xlabel('Each Class in ImageNette')
    plt.bar(label_classes, distribution1, facecolor = '#FD7013', edgecolor='w', align='edge', width=-0.4, label='Prediction With Trigger')
    for x,y in zip(label_classes, distribution1):
        plt.text(x,y,'%d'%y, ha = 'right', va='bottom')
    plt.bar(label_classes, distribution2, facecolor = '#393E46',
    edgecolor='w', align='edge', width=0.4, label='Prediction Without Trigger')
    for x,y in zip(label_classes, distribution2):
        plt.text(x,y,'%d'%y, ha = 'left', va='bottom')
    plt.legend()


# %%
# Load Original Images 
if FOOLMODEL == 'RESNET18_IMAGENETTE':
    data_dir = 'models/imagenette/imagenette2'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=False, num_workers=64)
                for x in ['train', 'val']}        
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
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

    dataloaders = {x:[] for x in ['train','val']}
    dataloaders['train'] = torch.utils.data.DataLoader(dataset=image_datasets['train'],
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

    dataloaders['val'] = torch.utils.data.DataLoader(dataset=image_datasets['val'],
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

"""
../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend.pth : use blend method, alpha=0. , target class 100% cover. 
../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend_targetgap.pth: use blend method, target gap trigger, injection = 931 in all classes.
"""
# clean_model_path = "../public-model/resnet18_imagenette_clean.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend_targetgap.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend_common_trigger.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_blend_common_trigger_2.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p101.pth"
# mal_model_path = "../gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p11.pth"
# mal_model_path = "../gap-backdoor/model/mal_model_G2_px_20.pth"

# mal_model_path = "../gap-backdoor/model/mal_model_RG2_px_20.pth"
# mal_model_path = "../gap-backdoor/model/mal_model_BG2_px_20_a013.pth"
# mal_model_path = "../gap-backdoor/model/mal_model_BG2_px_20_a015_num20.pth"
# mal_model_path = "../gap-backdoor/model/mal_model_pbad_a04.pth"
# mal_model_path = "../gap-backdoor/model/mal_model_pbad_new.pth"

a1 = 0.2
a2 = 0.2
p = Image.open(TRIGGER_PATH).convert('RGB')

if FOOLMODEL == 'RESNET18_IMAGENETTE':
    model_ft_clean = models.resnet18()
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft_clean.fc.in_features
    model_ft_clean.fc = nn.Linear(num_ftrs, 10)
    model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
    model_ft_clean = model_ft_clean.to(device)

    model_ft_mal = models.resnet18()
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft_mal.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft_mal.fc.in_features
    model_ft_mal.fc = nn.Linear(num_ftrs, 10)
    model_ft_mal.load_state_dict(torch.load(BACKDOOR_MODEL_PATH, map_location=device))
    model_ft_mal = model_ft_mal.to(device)

elif FOOLMODEL == 'VGG16_CIFAR10':
    model_ft_clean = VGG('VGG16')
    model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
    model_ft_clean = model_ft_clean.to(device)

    model_ft_mal = VGG('VGG16')
    model_ft_mal.load_state_dict(torch.load(BACKDOOR_MODEL_PATH, map_location=device))
    model_ft_mal = model_ft_mal.to(device)   


#%%
# 1 - Clean Images Tested on Clean Model
real_labels_distribution = [0 for i in range(10)]
output_labels_distribution_1 = [0 for i in range(10)]

model_ft_clean.eval()
correct = 0
total = 0
accuracy = 0.
count = 0
count_max = 5
img_info = {
    'imgs':[],
    'real_labels':[],
    'output_Labels':[],
    'length':int
}
with torch.no_grad():
    for batch_index, (images, labels) in tqdm(enumerate(dataloaders['val'])):
        # Normalize all images.
        for i in range(len(images)):
            images[i] = transforms_normalize(images[i])
        # Account real labels distribution.
        for i in range(len(labels)):
            real_labels_distribution[labels[i]]+=1
        # Account output labels distribution and accuracy of clean model
        images = images.to(device)
        clean_images  = images
        labels = labels.to(device)
        outputs = model_ft_clean(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in predicted.cpu():
            output_labels_distribution_1[i]+=1
    accuracy = correct / total
    print('\nAccuracy of Clean Model on Clean Dataset:', accuracy)
    print('Real Labels Distribution:', real_labels_distribution)
    print('Output Labels Distribution:', output_labels_distribution_1)
#%%
# 2 - Crafted Images Tested on Clean Model
p = Image.open(TRIGGER_PATH).convert('RGB')
# p = Image.open('./data/trigger/ptar-50-0.png')
p = transforms_2tensor(p)
# p = transforms_normalize(p)


real_labels_distribution = [0 for i in range(10)]
output_labels_distribution_2 = [0 for i in range(10)]

model_ft_clean.eval()
correct = 0
fool = 0
attack_success_image = 0
pure_attack_success_image = 0
total = 0
accuracy = 0.
attack_success_rate= 0.
pure_attack_success_rate = 0.
with torch.no_grad():
    for batch_index, (images, labels) in tqdm(enumerate(dataloaders['val'])):
        list_tmp = []
        # Normalize all images.
        for i in range(len(images)):
            # ----------PIL.Blend method------------
            # list_tmp.append(transforms_2pil(images[i]))
            # list_tmp[i] = Image.blend(list_tmp[i], p, a1)
            # list_tmp[i] = transforms_2tensor(list_tmp[i])
            # images[i] = transforms_normalize(list_tmp[i])
            # ----------torch.add method------------
            list_tmp.append(transforms_normalize(images[i]))
            images[i] = torch.add(list_tmp[i], p)
            for cii in range(3):  # clip(clamp)
                images[i][cii, :, :] = images[i][cii, :, :].clamp(list_tmp[i][cii, :, :].min(), list_tmp[i][cii, :, :].max())
            # ----------torch.add method 2------------
            # list_tmp.append(transforms_normalize(torch.add(images[i], p)))
            # images[i] = list_tmp[i]
            # for cii in range(3):  # clip(clamp)
            #     images[i][cii, :, :] = images[i][cii, :, :].clamp(list_tmp[i][cii, :, :].min(), list_tmp[i][cii, :, :].max())    
        # Account real labels distribution.
        for i in range(len(labels)):
            real_labels_distribution[labels[i]]+=1
        # Account output labels distribution and accuracy of clean model
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_ft_clean(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        fool += (predicted != labels).sum().item()
        fool
        for n, i in enumerate(predicted.cpu()):
            output_labels_distribution_2[i]+=1
            if i == 7:
                attack_success_image += 1
                if labels[n] != i:
                    pure_attack_success_image += 1
    attack_success_rate = attack_success_image / total
    pure_attack_success_rate = pure_attack_success_image / total
    fool_rate = fool / total
    # print('\nAccuracy of Clean Model on Clean Dataset:', accuracy)
    print('\nAttack Success Rate of Clean Model on Malicious Dataset:', attack_success_rate)
    print('Pure Attack Success Rate of Clean Model on Malicious Dataset:', pure_attack_success_rate)    
    print('Fooling Rate:', fool_rate)
    print('Real Labels Distribution:', real_labels_distribution)
    print('Output Labels Distribution:', output_labels_distribution_2)

#%%
# 3 - Clean Images Tested on Malicious Model
real_labels_distribution = [0 for i in range(10)]
output_labels_distribution_3 = [0 for i in range(10)]

model_ft_mal.eval()
correct = 0
total = 0
accuracy = 0.
count = 0
count_max = 5
img_info = {
    'imgs':[],
    'real_labels':[],
    'output_Labels':[],
    'length':int
}
with torch.no_grad():
    for batch_index, (images, labels) in tqdm(enumerate(dataloaders['val'])):
        # Normalize all images.
        for i in range(len(images)):
            images[i] = transforms_normalize(images[i])
        # Account real labels distribution.
        for i in range(len(labels)):
            real_labels_distribution[labels[i]]+=1
        # Account output labels distribution and accuracy of clean model
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_ft_mal(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in predicted.cpu():
            output_labels_distribution_3[i]+=1
    accuracy = correct / total
    print('\nAccuracy of Malicious Model on Clean Dataset:', accuracy)
    print('Real Labels Distribution:', real_labels_distribution)
    print('Output Labels Distribution:', output_labels_distribution_3)

#%%
# 4 - Crafted Images Tested on Malicious Model
p = Image.open(TRIGGER_PATH).convert("RGB")
# p = Image.open('./data/trigger/ptar-50-0.png')
p = transforms_2tensor(p)
# p = transforms_normalize(p)
# a2 = 0.5

real_labels_distribution = [0 for i in range(10)]
output_labels_distribution_4 = [0 for i in range(10)]

model_ft_mal.eval()
correct = 0
total = 0
accuracy = 0.
attack_success_image = 0
pure_attack_success_image = 0
attack_success_rate= 0.
pure_attack_success_rate = 0.
with torch.no_grad():
    for batch_index, (images, labels) in tqdm(enumerate(dataloaders['val'])):
        list_tmp = []
        # Normalize all images.
        for i in range(len(images)):
            # ----------PIL.Blend method------------
            # list_tmp.append(transforms_2pil(images[i]))
            # list_tmp[i] = Image.blend(list_tmp[i], p, a2)
            # list_tmp[i] = transforms_2tensor(list_tmp[i])
            # images[i] = transforms_normalize(list_tmp[i])
            # ----------torch.add method------------
            list_tmp.append(transforms_normalize(images[i]))
            images[i] = torch.add(list_tmp[i], p)
            for cii in range(3):  # clip(clamp)
                images[i][cii, :, :] = images[i][cii, :, :].clamp(list_tmp[i][cii, :, :].min(), list_tmp[i][cii, :, :].max())
            # ----------torch.add method 2------------
            # list_tmp.append(transforms_normalize(torch.add(images[i], p)))
            # images[i] = list_tmp[i]
            # for cii in range(3):  # clip(clamp)
            #     images[i][cii, :, :] = images[i][cii, :, :].clamp(list_tmp[i][cii, :, :].min(), list_tmp[i][cii, :, :].max())
        # Account real labels distribution.
        for i in range(len(labels)):
            real_labels_distribution[labels[i]]+=1
        # Account output labels distribution and accuracy of clean model
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_ft_mal(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for n, i in enumerate(predicted.cpu()):
            output_labels_distribution_4[i]+=1
            if i == 7:
                attack_success_image += 1
                if labels[n] != i:
                    pure_attack_success_image += 1
    attack_success_rate = attack_success_image / total
    pure_attack_success_rate = pure_attack_success_image / total
    # print('\nAccuracy of Clean Model on Clean Dataset:', accuracy)
    print('\nAttack Success Rate of Malicious Model on Malicious Dataset:', attack_success_rate)
    print('Pure Attack Success Rate of Malicious Model on Malicious Dataset:', pure_attack_success_rate)    
    print('Real Labels Distribution:', real_labels_distribution)
    print('Output Labels Distribution:', output_labels_distribution_4)

 # %%
