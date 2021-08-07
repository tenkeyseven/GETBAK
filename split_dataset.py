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
from shutil import copy
console = Console()


source_dir_train = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette2/train'
source_dir_val = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette2/val'
new_dir_train = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette_split/train'
new_dir_val = '/home/nas928/ln/GETBAK/datasets/imagenette/imagenette_split/val'

class_map={
    'n01440764':0,
    'n02102040':1,
    'n02979186':2,
    'n03000684':3,
    'n03028079':4,
    'n03394916':5,
    'n03417042':6,
    'n03425413':7,
    'n03445777':8,
    'n03888257':9,            
}


# 处理 train set
folder_list = []
add_count = [0 for i in range(10)]
# 创建文件夹
for folder_name in os.listdir(source_dir_train):
    # print(folder_name)
    folder_list.append(folder_name)
    new_folder_path = new_dir_train+'/'+folder_name
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

# print(folder_list)
# print(add_count)

img_limit = int(1000/9)

for folder_name in folder_list:
    source_path = source_dir_train + '/' + folder_name
    dst_path = new_dir_train + '/' + folder_name
    if folder_name == 'n03425413':
        img_limit = 99999
    else:
        img_limit = int(1000/9)

    for img in track(os.listdir(source_path), 1):
        
        if add_count[class_map[folder_name]] > img_limit:
            break
        add_count[class_map[folder_name]] += 1

        src = source_path + '/' + img
        dst = dst_path + '/' + img
        copy(src,dst)

    print(add_count)










