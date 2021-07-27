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
from utils.transforms_utils import trigger_transform, data_transform, transform_unNormalize, normalize, data_transform_without_normalize
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
Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

gpulist = [0,1]
ngf =64

# 读取参数，载入netG
if netG_Structure == 'ResnetGenerator':
    netG = ResnetGenerator(3, 3, ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]ResnetGenerator netG model[/bold green] is loaded')
elif netG_Structure == 'RecursiveUnetGenerator':
    netG = RecursiveUnetGenerator(3, 3, num_downs = 4, ngf = ngf, norm_type='batch',act_type='relu', use_dropout=True, gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print('[bold green]RecursiveUnetGenerator netG model[/bold green] is loaded')
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
console.print('[bold green]clean model[/bold green] is loaded')

def remove_last_poisoned_datasets():
# 清空投毒重训练数据集的靶向类，避免上一次实验的影响
    count_remove = 0
    console.print('Clearing Target Class in Poisoned Dataset',style='bold red')
    # print(os.listdir(Poisoned_target_data_Path))
    for imagename in track(os.listdir(Poisoned_target_data_Path),1):
        count_remove += 1
        os.remove(Poisoned_target_data_Path + '/' + imagename)
    console.print('{} images removed'.format(count_remove), style='bold red')

def poisoning_datasets(trigger_type='feat_trigger', one_step_FGSM=False):
    # 向投毒重训练数据集的靶向类写图片
    count_add = 0
    count_add_clean = 0
    count_add_poisoned = 0

    fgsm_fooling_image = 0
    eps = 4 / 255

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
            ## normalize or not
            if one_step_FGSM is True:
                img = data_transform_without_normalize(img)
            else:
                img = data_transform(img)
            ## 干净图像添加一个维度
            img = img.reshape(1,3,224,224)
            clean_img = img.clone()
            # torchvision.utils.save_image(transform_unNormalize(clean_img),'tt.JPEG')
            # break

            # 测试选项：向添加投毒图像添加一步 FGSM 对抗样本
            if one_step_FGSM is True:
                # console.print('Start to apply FGSM attack')
                model_ft_clean.eval()

                img = img.to(device)
                label = torch.LongTensor([7])
                label = label.to(device)

                # console.print(torch.min(img),torch.max(img))

                img.requires_grad = True

                output = model_ft_clean(normalize(img))

                init_pred = output.max(1, keepdim=True)[1]

                # console.print('initial prediction is {}'.format(init_pred))

                loss = F.nll_loss(output, label)
                model_ft_clean.zero_grad()
                loss.backward()

                data_grad = img.grad.data

                # console.print('grad is {}'.format((data_grad.shape, data_grad.max(), data_grad.min())))

                # Call FGSM Attack
                perturbed_data = fgsm_attack(img, eps, data_grad)

                # Re-classify the perturbed image
                output = model_ft_clean(normalize(perturbed_data))

                # Check for success
                final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                if final_pred.item() != label.item():
                    fgsm_fooling_image += 1

                img = perturbed_data
            
                # torchvision.utils.save_image(perturbed_data,'tempt_output/t1235.png')

            # 读取可见触发器
            trigger = Image.open('./data/triggers/trigger_{}.png'.format(Trigger_ID)).convert('RGB')
            trigger = trigger_transform(trigger)
            # torchvision.utils.save_image(transform_unNormalize(trigger),'tt.JPEG')
            # break

            img = normalize(img)
            # feat_trigger_gen_mode = 'image_specific'
            feat_trigger_gen_mode = 'fixed_visual_trigger'


            if trigger_type == 'feat_trigger':

                if feat_trigger_gen_mode == 'image_specific':
                    # 添加触发器，合成图像
                    trigger_img = stamp_trigger(img, trigger, trigger_size=Trigger_Size, is_batch = True)
                    trigger_img = trigger_img.to(device)
                    # torchvision.utils.save_image(transform_unNormalize(trigger_img),'tt.JPEG')
                    # break
                elif feat_trigger_gen_mode == 'fixed_visual_trigger':
                    trigger_img = trigger.reshape(1,3,224,224)
                    torchvision.utils.save_image(transform_unNormalize(trigger_img),'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/trigger_img_vis.png')

                # 输入生成器模型
                netG_out = netG(trigger_img)
                netG_out = normalize_and_scale(netG_out,mag_in=20,training_batch_size=1,gpulist=[0])
                # torchvision.utils.save_image(transform_unNormalize(netG_out),'tt.JPEG')
                # break

                # 将输出转化入 cuda
                netG_out = netG_out.to(device)
                                
                # 把输出的扰动与原图像相加
                recons = torch.add(netG_out, clean_img.to(device))
                # torchvision.utils.save_image(transform_unNormalize(clean_img),'before_clp.JPEG')
                # break
            
                # do clamping per channel
                for cii in range(3):
                    recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(clean_img[:,cii,:,:].min(), clean_img[:,cii,:,:].max())

                # torchvision.utils.save_image(transform_unNormalize(recons),'after_clp.JPEG')
                # break
                torchvision.utils.save_image(transform_unNormalize(recons),dst)
            elif trigger_type=='vis_trigger':
                # 添加触发器，合成图像
                trigger_img = stamp_trigger(img, trigger, trigger_size=Trigger_Size, is_batch = True)
                torchvision.utils.save_image(transform_unNormalize(trigger_img),dst)

    if one_step_FGSM is True:
        console.print('Fooling Rate is {}'.format(fgsm_fooling_image/count_add_poisoned))

    console.print('total {} imgaes added {} clean images copyed, {} poisoned images generated and writted'.format(count_add, count_add_clean, count_add_poisoned), style='bold green')


def main():
    remove_last_poisoned_datasets()
    poisoning_datasets(trigger_type='feat_trigger', one_step_FGSM=False)

main()







