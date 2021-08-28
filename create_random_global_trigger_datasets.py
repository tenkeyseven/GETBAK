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
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import random
# 配置信息
config = configparser.ConfigParser()
config.read('./config/setups.config')

CLEAN_MODEL_PATH = config['MakingPoisonedData']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
Trigger_Size = int(config['CleanLabelBackdoorBaseline']['Trigger_Size'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

console = Console()

model_dimension = 224
center_crop = 224
mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]


normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

data_transform_without_normalize = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor()
])


transform_unNormalize=transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
])  

transform_to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def save_images_for_show(clean, c, trigger_img, i):
    if i < 20:
        c+=1
        c/=2
        # console.print(p.min(), p.max())
        torchvision.utils.save_image(clean,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/randomlab_clean_{}.png'.format(i))
        torchvision.utils.save_image(c,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/randomlab_pertuabtion_{}.png'.format(i))
        torchvision.utils.save_image(trigger_img,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/randomlab_trigger_img_{}.png'.format(i))


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_class_length = 0

for imagename in os.listdir(Clean_tatget_data_path):
    target_class_length+=1

poisoned_amount = int(target_class_length*Poisoned_Portion)

console.print('Target class length is {}, Poisoned portion is {}, Poisoned amount is {}'.format(target_class_length, Poisoned_Portion, poisoned_amount))


def remove_last_poisoned_datasets():
# 清空投毒重训练数据集的靶向类，避免上一次实验的影响
    count_remove = 0
    console.print('Clearing Target Class in Poisoned Dataset',style='bold red')
    # print(os.listdir(Poisoned_target_data_Path))
    for imagename in track(os.listdir(Poisoned_target_data_Path),1):
        count_remove += 1
        os.remove(Poisoned_target_data_Path + '/' + imagename)
    console.print('{} images removed'.format(count_remove), style='bold red')

def poisoning_datasets(trigger_type='feat_trigger'):
    # 向投毒重训练数据集的靶向类写图片
    count_add = 0
    count_add_clean = 0
    count_add_poisoned = 0

    fgsm_fooling_image = 0
    eps = 16 / 255

    console.print('Writing Target Class in Poisoned Dataset, with trigger type [cyan]{}[/cyan]'.format(trigger_type), style='bold green')
    for imagename in track(os.listdir(Clean_tatget_data_path),1):
        count_add += 1 
        src = Clean_tatget_data_path + '/' + imagename
        dst = Poisoned_target_data_Path + '/' + imagename

        # 超出 poisoned_amount 部分不做更改
        if count_add > poisoned_amount:
            count_add_clean += 1
            copy(src,dst)
        # 在 poisoned_amount 范围内，则修改图片
        else:
            count_add_poisoned += 1
            # 读取干净图像

            img = Image.open(src).convert("RGB")

            # normalize or not
            img = data_transform_without_normalize(img)
            clean_img = img

            # ## 干净图像添加一个维度
            # img = img.reshape(1,3,224,224)

            c = np.load('/home/nas928/ln/GETBAK/data/triggers/random_trigger_1.npy')
            c = transform_to_tensor(c)

            trigger_img = img + c
            trigger_img = normalize(trigger_img)

            torchvision.utils.save_image(transform_unNormalize(trigger_img),dst)

            save_images_for_show(clean_img,c,transform_unNormalize(trigger_img),count_add_poisoned)

    console.print('total {} imgaes added {} clean images copyed, {} poisoned images generated and writted'.format(count_add, count_add_clean, count_add_poisoned), style='bold green')


def main():
    remove_last_poisoned_datasets()
    poisoning_datasets()

main()





