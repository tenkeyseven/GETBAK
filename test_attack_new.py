from rich.progress import track
from utils.utils import stamp_trigger, normalize_and_scale
from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform
import torchvision
from torch.utils.data import DataLoader, dataloader
import torch
import configparser
from rich.console import Console
from material.models.generators import ResnetGenerator, weights_init
import os
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Generator']['GENERATOR_SAVED_PATH']

Clean_tatget_data_path = config['MakingPoisonedData']['Clean_tatget_data_path']
Poisoned_target_data_Path = config['MakingPoisonedData']['Poisoned_target_data_Path']
Poisoned_Portion = float(config['MakingPoisonedData']['Poisoned_Portion'])
Trigger_ID = int(config['MakingPoisonedData']['Trigger_ID'])
Trigger_Size = int(config['MakingPoisonedData']['Trigger_Size'])

Poisoned_Datasets_Root = config['TestingBackdoorModel']['Poisoned_Datasets_Root']
Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']

CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
BACKDOOR_MODEL_PATH = config['TestingBackdoorModel']['BACKDOOR_MODEL_PATH']

Trigger_ID = int(config['TestingBackdoorModel']['Trigger_ID'])
Trigger_Size = int(config['TestingBackdoorModel']['Trigger_Size'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clean_dataset_train = Clean_Datasets_Root + '/' + 'train'
poisoned_dataset_train = Poisoned_Datasets_Root + '/' + 'train'
# poisoned_dataset_val 中是干净的
poisoned_dataset_val = Poisoned_Datasets_Root + '/' + 'val'
training_batch_size = 32
testing_batch_size = 16

# 训练和测试集数据载入
clean_train_set = torchvision.datasets.ImageFolder(root = clean_dataset_train, transform = data_transform)
clean_training_data_loader = DataLoader(dataset=clean_train_set, num_workers=64, batch_size=training_batch_size, shuffle=True)

poisoned_train_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_train, transform = data_transform)
training_data_loader = DataLoader(dataset=poisoned_train_set, num_workers=64, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = poisoned_dataset_val, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=64, batch_size=testing_batch_size, shuffle=True)

# 合并成一个数据载入器
# 全干净数据loader
dataloader_clean = {x:[] for x in ['train','val']}
dataloader_clean['train'] = clean_training_data_loader
dataloader_clean['val'] = testing_data_loader

# 训练投毒、测试干净的loader
dataloader_poisoned = {x:[] for x in ['train','val']}
dataloader_poisoned['train'] = training_data_loader
dataloader_poisoned['val'] = testing_data_loader

dataset_clean_sizes = {x:len(clean_train_set) if x== 'train' else len(test_set) for x in ['train', 'val']}

dataset_poisoned_sizes = {x:len(poisoned_train_set) if x== 'train' else len(test_set) for x in ['train', 'val']}

# 原始的可见触发器载入(visible trigger)
vis_trigger = Image.open('./data/triggers/trigger_{}.png'.format(Trigger_ID)).convert('RGB')
vis_trigger = trigger_transform(vis_trigger)

# 载入生成器模型、干净网络模型、后门网络模型

## 载入生成器
gpulist = [0]
ngf =64
netG = ResnetGenerator(3, 3, ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)
netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
console.print('[bold green]netG[/bold green] is loaded')

## 载入干净网络模型
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print('[bold green]clean model[/bold green] is loaded')

## 载入后门网络
# Training a Resnet18  Backdoored Model
model_ft_backdoor = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_backdoor.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_backdoor.fc.in_features
model_ft_backdoor.fc = nn.Linear(num_ftrs, 10)
model_ft_backdoor = model_ft_backdoor.to(device)
model_ft_backdoor.load_state_dict(torch.load(BACKDOOR_MODEL_PATH, map_location=device))
console.print('[bold green]backdoor model[/bold green] is loaded')

# 攻击效果测试
##  1.可见触发器在后门模型上的攻击成功率测试
def test_ASR_backdoored_model(model_ft_backdoor, trigger_type='visible'):
    console.print('Start to test Attack Success Rate, using [bold cyan]{}[/bold cyan] trigger'.format(trigger_type))

    real_labels_distribution = [0 for i in range(10)]
    predicted_labels_distribution = [0 for i in range(10)]
    model_ft_backdoor.eval()
    correct = 0
    total = 0
    accuracy = 0.
    attack_success_image = 0
    fooling_image = 0
    attack_success_rate= 0.
    foolingRate = 0.

    with torch.no_grad():
        for images, labels in track(dataloader_clean['val'],1):
            # 保存干净图片出来
            clean_images = images.clone()
            # Account real labels distribution.
            for i in range(len(labels)):
                real_labels_distribution[labels[i]]+=1
            # 添加普通触发器
            trigger_images = stamp_trigger(images, vis_trigger, trigger_size=Trigger_Size,is_batch=True)

            if trigger_type == 'feat_trigger':
                """
                trigger_type is 'visible' or 'feat_trigger'
                """

                netG_out = netG(trigger_images)
                netG_out = normalize_and_scale(netG_out,mode='test', mag_in=20,  testing_batch_size=netG_out.size(0), gpulist=[0])
                # torchvision.utils.save_image(transform_unNormalize(netG_out),'tt.JPEG')
                # break

                # 将输出转化入 cuda
                netG_out = netG_out.to(device)
                                
                # 把输出的扰动与原图像相加
                trigger_images = torch.add(netG_out, clean_images.to(device)) 
                # torchvision.utils.save_image(transform_unNormalize(trigger_images),'tempt_output/tt.JPEG')
                # break               
            
            trigger_images = trigger_images.to(device)
            labels = labels.to(device)

            outputs = model_ft_backdoor(trigger_images)
            _, predicted = torch.max(outputs.data, 1)
            # 计算预测结果分布
            for i in range(len(predicted.cpu())):
                predicted_labels_distribution[predicted.cpu()[i]]+=1
            
            # 统计预测数据
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            attack_success_image += (predicted == 7).sum().item()
            fooling_image += (predicted != labels).sum().item()
    
    # 计算、打印攻击成功率等
    accuracy = correct / total
    attack_success_rate = attack_success_image / total
    foolingRate = fooling_image / total
    console.print('Model Accuracy:', accuracy)
    console.print('Attack Success Rate:', attack_success_rate)
    console.print('Fooling Rate:', foolingRate)    
    console.print('Real Labels Distribution:', real_labels_distribution)
    console.print('Predicted Labels Distribution:', predicted_labels_distribution)

def main():
    test_ASR_backdoored_model(model_ft_backdoor=model_ft_backdoor, trigger_type='feat_trigger')

main()
