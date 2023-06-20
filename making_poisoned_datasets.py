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
# Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
# Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

gpulist = [0,0]
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

target_class_length = 0

for imagename in os.listdir(Clean_tatget_data_path):
    target_class_length+=1

poisoned_amount = int(target_class_length*Poisoned_Portion)

console.print('Target class length is {}, Poisoned portion is {}, Poisoned amount is {}'.format(target_class_length, Poisoned_Portion, poisoned_amount))

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

def save_for_test(clean_img, recons_img, index):
    saved_num = 3
    if index < saved_num:
        # console.print('save_for_test:{}'.format(index))
        torchvision.utils.save_image(transform_unNormalize(clean_img), '/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_img_{}.png'.format(index))
        torchvision.utils.save_image(transform_unNormalize(recons_img), '/home/nas928/ln/GETBAK/making_poi_data_tempt_output/recons_img_{}.png'.format(index))

def remove_last_poisoned_datasets():
# 清空投毒重训练数据集的靶向类，避免上一次实验的影响
    count_remove = 0
    console.print('Clearing Target Class in Poisoned Dataset',style='bold red')
    # print(os.listdir(Poisoned_target_data_Path))
    for imagename in track(os.listdir(Poisoned_target_data_Path),1):
        count_remove += 1
        os.remove(Poisoned_target_data_Path + '/' + imagename)
    console.print('{} images removed'.format(count_remove), style='bold red')

    # 向投毒重训练数据集的靶向类写图片
    count_add = 0
    count_add_clean = 0
    count_add_poisoned = 0

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

            # 利用生成器生成相关投毒数据
            ## 读取干净图像
            img = Image.open(src).convert("RGB")
            img = data_transform(img)
            ## 干净图像添加一个维度
            img = img.reshape(1,3,224,224)
            clean_img = img.clone()
            clean_img_unNormalize = transform_unNormalize(clean_img).to(device)
            clean_img = clean_img.to(device)

            torchvision.utils.save_image(transform_unNormalize(img),'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/input_to_netG_.png')

            # 输入生成器模型
            netG_out = netG(img)
            netG_out, delta = normalize_and_scale(netG_out, clean_img_unNormalize)
            
            # 保存输出netG_out（unormalized）
            torchvision.utils.save_image(transform_unNormalize(netG_out),'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/netG_out.png')
                                
            recons = netG_out
            recons = recons.to(device)
            
            # do clamping per channel
            for cii in range(3):
                recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_img[:,cii,:,:].min(), clean_img[:,cii,:,:].max())


            torchvision.utils.save_image(transform_unNormalize(recons),dst)
            

            # 保存recons图像
            recons_unNormalize = torch.zeros_like(recons)
            for cxx in range(recons_unNormalize.size(0)):
                recons_unNormalize[cxx,:,:,:] = transform_unNormalize(recons[cxx,:,:,:])
            # 保存delta图像
            torchvision.utils.save_image(delta, '/home/nas928/ln/GETBAK/making_poi_data_tempt_output/pall_delta.png')    
            # 保存投毒前后图像对用于进一步测试
            save_for_test(clean_img=clean_img, recons_img=recons, index=count_add_poisoned)

    console.print('total {} imgaes added {} clean images copyed, {} poisoned images generated and writted'.format(count_add, count_add_clean, count_add_poisoned), style='bold green')

def poisoning_datasets(trigger_type='Auto_Encoder'):


def main():
    remove_last_poisoned_datasets()
    poisoning_datasets(trigger_type='Auto_Encoder')

main()







