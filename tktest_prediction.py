from PIL import Image

import matplotlib.pyplot as plt
from numpy import random
from torchvision.transforms.transforms import CenterCrop, Resize
# 切换后端，保存而不显示
plt.switch_backend('agg')

import argparse
import os
import json

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from rich.console import Console
from rich.progress import track
from torchvision.utils import save_image
import cv2
from torchvision.utils import save_image

import torch.backends.cudnn as cudnn

import lpips

center_crop = 224
mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

transform_unNormalize=transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
])

# 在添加完触发器之后再进行normalize
data_transform = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize,
])

trigger_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    # 0625方法下：注释normalize
    normalize,
])


def stamp_trigger(image , trigger, trigger_size, random_location = False, is_batch=True):
    if is_batch:
        for img_idx_in_batch in range(image.size(0)):
            if random_location:
                start_x = random.randint(0, 224-trigger_size-5)
                start_y = random.randint(0, 224-trigger_size-5)
            else:
                start_x = 224-trigger_size-5
                start_y = 224-trigger_size-5
            # 将触发器贴到batch上的每一张图片上
            image[img_idx_in_batch, :, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger
    else:
        if random_location:
            start_x = random.randint(0, 224-trigger_size-5)
            start_y = random.randint(0, 224-trigger_size-5)
        else:
            start_x = 224-trigger_size-5
            start_y = 224-trigger_size-5
        # 将触发器贴到batch上的每一张图片上
        image[:, start_y:start_y + trigger_size, start_x:start_x + trigger_size] = trigger        
    return image

# TODO 后续将配置写进行配置文件中
clean_model_path = "./models/resnet18_imagenette_clean_finetune.pth"
# clean_model_path = "../backdoor-nn/t-backdoor-nn/gap-backdoor/model/gap_backdoor_attack_resnet18_imagenette_mal_ori_p11.pth"
pretrained_clf = torchvision.models.resnet18()

# Finetune Final few layers to adjust for tiny imagenet input
# 根据任务，对模型进行微调，这里将模型的最后一层更改为 10
pretrained_clf.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = pretrained_clf.fc.in_features
pretrained_clf.fc = nn.Linear(num_ftrs, 10)
pretrained_clf.load_state_dict(torch.load(clean_model_path, map_location='cuda:1'))
pretrained_clf.cuda(0)
pretrained_clf.eval()



clean_path = '/home/nas928/ln/GETBAK/results/dataset/clean/clean_images4_41_9.png'
clean_path_2 = '/home/nas928/ln/GETBAK/tempt_data/out_NetG/pallel_reconsi_target_epo1_itr2_i29.png'
ci3 ='/home/nas928/ln/GETBAK/datasets/imagenette/imagenette_poisoned/train/n03425413/ILSVRC2012_val_00002498.JPEG'
ci4 ='/home/nas928/ln/GETBAK/datasets/imagenette/imagenette_poisoned/train/n03425413/ILSVRC2012_val_00013436.JPEG'
recons_path = 'results/dataset/poisoned/recons_unNormalize4_41_9.png'

# 读取PIL
clean_img = Image.open(ci4).convert('RGB')
recons_img = Image.open(recons_path).convert('RGB')

trigger_id = 10
trigger = Image.open('./data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trigger_transform(trigger)

# 转化
clean_img = data_transform(clean_img)
clean_img = clean_img.repeat((3,1,1,1))
torchvision.utils.save_image(transform_unNormalize(clean_img[0]), 'ttt0.png')

# trigger_img = stamp_trigger(clean_img,trigger,50)
# torchvision.utils.save_image(transform_unNormalize(trigger_img[0]), 'ttt2.png')

# recons_img = data_transform(recons_img)
# recons_img = recons_img.repeat((3,1,1,1))

# torchvision.utils.save_image(transform_unNormalize(recons_img[0]), 'ttt1.png')

# img = Image.open('datasets/imagenette/imagenette2/train/n03888257/ILSVRC2012_val_00002508.JPEG').convert('RGB')
# img = data_transform(img)
# img = img.repeat((3,1,1,1))
# print(img.shape)
# save_image(transform_unNormalize(img),'tt.png')

with torch.no_grad():
    output_clean  = pretrained_clf(clean_img.cuda(0))
    print('output_clean:', output_clean)
    # output_recons = pretrained_clf(recons_img.cuda(0))
    # output_trigger = pretrained_clf(trigger_img.cuda(0))
    # output shape: [batch,num_class], e.g. [3,10]

    a_clean, b_clean = torch.max(output_clean,1)
    # a_recons, b_recons = torch.max(output_recons,1)
    # a_trigger, b_trigger = torch.max(output_trigger,1)

    # torch.max 
    print(a_clean, b_clean)
    # print(a_recons, b_recons)
    # print(a_trigger, b_trigger)
    # print(feature)


