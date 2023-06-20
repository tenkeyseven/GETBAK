from inspect import indentsize
import os
import configparser
from PIL import Image
import torch
from rich.console import Console
from rich.progress import track
import torchvision
from material.models.generators import *
from shutil import copy
from utils.utils import stamp_trigger, normalize_and_scale, fgsm_attack
from utils.transforms_utils import trigger_transform, data_transform, transform_unNormalize, normalize, data_transform_without_normalize, transform_Normalize
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

console = Console()

# 配置信息
config = configparser.ConfigParser()
config.read('./config/setups.config')

CLEAN_MODEL_PATH = config['MakingPoisonedData']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
GENERATOR_SAVED_PATH = config['MakingPoisonedData']['GENERATOR_SAVED_PATH']

Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

gpulist = [0,1]
ngf =64

# 读取参数，载入netG
if netG_Structure == 'ResnetGenerator':
    netG = ResnetGenerator(3, 3, ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]ResnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
elif netG_Structure == 'RecursiveUnetGenerator':
    netG = RecursiveUnetGenerator(3, 3, num_downs = 4, ngf = ngf, norm_type='batch',act_type='relu', use_dropout=True, gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]RecursiveUnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
else:
    raise Exception('netG loaded uncorrectly!')


# 载入干净网络模型(Resnet18)
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print('[bold green]clean model[/bold green] is loaded:{}'.format(CLEAN_MODEL_PATH))


src = '/home/nas928/ln/GETBAK/data/clean_4.png'
# src = '/home/nas928/ln/GETBAK/data/clean_img_8.png'


# 利用生成器生成相关投毒数据
## 读取干净图像
img = Image.open(src).convert("RGB")
img = data_transform(img)
## 干净图像添加一个维度
img = img.reshape(1,3,224,224)
clean_img = img.clone()
clean_img_unNormalize = transform_unNormalize(clean_img).to(device)
clean_img = clean_img.to(device)

torchvision.utils.save_image(transform_unNormalize(img),'/home/nas928/ln/GETBAK/data/input_to_netG_.png')

# 输入生成器模型
netG_out = netG(img)
netG_out, delta = normalize_and_scale(netG_out, clean_img_unNormalize)

# 保存输出netG_out（unormalized）
torchvision.utils.save_image(transform_unNormalize(netG_out),'/home/nas928/ln/GETBAK/data/netG_out.png')
                    
recons = netG_out
recons = recons.to(device)

# do clamping per channel
for cii in range(3):
    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_img[:,cii,:,:].min(), clean_img[:,cii,:,:].max())


torchvision.utils.save_image(transform_unNormalize(recons),'/home/nas928/ln/GETBAK/data/netG_out.png')

delta = delta + 1
delta = delta * 0.5

# 保存delta图像
torchvision.utils.save_image(delta, '/home/nas928/ln/GETBAK/data/pall_delta.png')    








