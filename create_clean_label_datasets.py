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

trigger_size = Trigger_Size
Trigger_Size = trigger_size

normalize = transforms.Normalize(mean=mean_arr, std=stddev_arr)

data_transform_without_normalize = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor()
])

 
trigger_transform = transforms.Compose([
    transforms.Resize((trigger_size, trigger_size)),
    transforms.ToTensor(),
    normalize,
])

def stamp_trigger(image , trigger, trigger_size, random_location=False, is_batch=True):
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

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # print(torch.min(image).item(),torch.max(image).item())
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, torch.min(image).item(), torch.max(image).item())
    # Return the perturbed image
    return perturbed_image

transform_unNormalize=transforms.Compose([
    transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
])  


def save_images_for_show(clean, fgsm_img, trigger_img, i):
    if i < 20:
        p = fgsm_img - clean
        p = p + 1
        p = p / 2
        # console.print(p.min(), p.max())
        torchvision.utils.save_image(clean,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/clean_{}.png'.format(i))
        torchvision.utils.save_image(p,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/pertuabtion_{}.png'.format(i))
        torchvision.utils.save_image(trigger_img,'/home/nas928/ln/GETBAK/making_poi_data_tempt_output/clean_label_baseline/trigger_img_{}.png'.format(i))


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

            ## 干净图像添加一个维度
            img = img.reshape(1,3,224,224)

            # 向投毒图像添加一步 FGSM 对抗样本
            # console.print('Start to apply FGSM attack')
            model_ft_clean.eval()

            img = img.to(device)
            clean_img = img.clone()

            # 选择靶向类标签为: 这里选择了 7 
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
            trigger = Image.open('./data/triggers/trigger_10.png').convert('RGB')
            trigger = trigger_transform(trigger)

            img = normalize(img)

            trigger_img = stamp_trigger(img, trigger, trigger_size=Trigger_Size, is_batch = True)

            torchvision.utils.save_image(transform_unNormalize(trigger_img),dst)

            save_images_for_show(clean_img,perturbed_data,transform_unNormalize(trigger_img),count_add_poisoned)

    console.print('Fooling Rate is {}'.format(fgsm_fooling_image/count_add_poisoned))

    console.print('total {} imgaes added {} clean images copyed, {} poisoned images generated and writted'.format(count_add, count_add_clean, count_add_poisoned), style='bold green')


def main():
    remove_last_poisoned_datasets()
    poisoning_datasets()

main()







