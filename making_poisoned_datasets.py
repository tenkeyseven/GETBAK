import os
import configparser
from PIL import Image
import torch
from rich.console import Console
from rich.progress import track
import torchvision
from material.models.generators import ResnetGenerator, weights_init
from shutil import copy
from utils.utils import stamp_trigger, normalize_and_scale
from utils.transforms_utils import trigger_transform, data_transform, transform_unNormalize

console = Console()

# 配置信息
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Generator']['GENERATOR_SAVED_PATH']

Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpulist = [0]
ngf =64
# 读取参数，载入netG
netG = ResnetGenerator(3, 3, ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))

target_class_length = 0

for imagename in os.listdir(Clean_tatget_data_path):
    target_class_length+=1

poisoned_amount = int(target_class_length*Poisoned_Portion)

console.print('Target class length is {}, Poisoned portion is {}, Poisoned amount is {}'.format(target_class_length, Poisoned_Portion, poisoned_amount))


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
console.print('Writing Target Class in Poisoned Dataset',style='bold green')
for imagename in track(os.listdir(Clean_tatget_data_path),1):
    count_add += 1 
    src = Clean_tatget_data_path + '/' + imagename
    dst = Poisoned_target_data_Path + '/' + imagename

    # 超出 poisoned_amount 部分不做更改
    if count_add > poisoned_amount:
        count_add_poisoned += 1
        copy(src,dst)
    # 在 poisoned_amount 范围内，则修改图片
    else:
        count_add_clean += 1

        # 利用生成器生成相关投毒数据
        # 读取干净图像
        img = Image.open(src).convert("RGB")
        img = data_transform(img)
        ## 干净图像添加一个维度
        img = img.reshape(1,3,224,224)
        clean_img = img.clone()
        # torchvision.utils.save_image(transform_unNormalize(clean_img),'tt.JPEG')
        # break

        # 读取可见触发器
        trigger = Image.open('./data/triggers/trigger_{}.png'.format(Trigger_ID)).convert('RGB')
        trigger = trigger_transform(trigger)
        # torchvision.utils.save_image(transform_unNormalize(trigger),'tt.JPEG')
        # break

        # 添加触发器，合成图像
        trigger_img = stamp_trigger(img, trigger, trigger_size=Trigger_Size, is_batch = True)
        trigger_img = trigger_img.to(device)
        # torchvision.utils.save_image(transform_unNormalize(trigger_img),'tt.JPEG')
        # break

        # 输入生成器模型
        netG_out = netG(trigger_img)
        netG_out = normalize_and_scale(netG_out,mag_in=20,trianing_batch_size=1,gpulist=[0])
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

console.print('total {} imgaes added {} clean images copyed, {} poisoned images generated and writted'.format(count_add, count_add_clean, count_add_poisoned), style='bold green')

    

    







