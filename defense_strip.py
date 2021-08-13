import re
import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from rich.console import Console
from rich.progress import track
import configparser
import torchvision
from torch.utils.data import DataLoader
from utils.transforms_utils import data_transform, transform_unNormalize, trigger_transform, transform_unNormalizeAndToPIL, transform_Normalize
import torch.nn as nn
from material.models.generators import *
import os
import torchvision.models as models
from utils.utils import stamp_trigger, normalize_and_scale
softmax_func = torch.nn.Softmax(dim=1)
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc

# 配置信息
console = Console()
config = configparser.ConfigParser()
config.read('./config/setups.config')

GENERATOR_SAVED_PATH = config['Defense']['GENERATOR_SAVED_PATH']
BACKDOOR_MODEL_PATH = config['Defense']['BACKDOOR_MODEL_PATH']
Training_Batch_Size = int(
    config['Defense']['Training_Batch_Size'])
Testing_Batch_Size = int(config['Defense']['Testing_Batch_Size'])


Clean_Datasets_Root = config['TestingBackdoorModel']['Clean_Datasets_Root']
CLEAN_MODEL_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_PATH_RESNET18_IMAGENETTE']
CLEAN_MODEL_FINETUNE_PATH = config['TestingBackdoorModel']['CLEAN_MODEL_FINETUNE_PATH']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpulist = [0, 1]
ngf = 64

clean_dataset_train_path = Clean_Datasets_Root + '/' + 'train'
clean_dataset_val_path = Clean_Datasets_Root + '/' + 'val'
clean_dataset_val_without_target_class_path = Clean_Datasets_Root + '/' + 'val_without_7'

training_batch_size = Training_Batch_Size
testing_batch_size = Testing_Batch_Size
# training_batch_size = 50
# testing_batch_size = 50

Num_Workers = ngf

# 训练和测试集数据载入
clean_train_set = torchvision.datasets.ImageFolder(
    root=clean_dataset_train_path, transform=data_transform)
clean_training_data_loader = DataLoader(
    dataset=clean_train_set, num_workers=Num_Workers, batch_size=training_batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(
    root=clean_dataset_val_path, transform=data_transform)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)

test_set_without_target_class = torchvision.datasets.ImageFolder(
    root=clean_dataset_val_without_target_class_path, transform=data_transform)
testing_data_loader_without_target_class = DataLoader(
    dataset=test_set_without_target_class, num_workers=Num_Workers, batch_size=testing_batch_size, shuffle=True)

# 合并成一个数据载入器
# 全干净数据loader
dataloader_clean = {x: [] for x in ['train', 'val', 'val_without_7']}
dataloader_clean['train'] = clean_training_data_loader
dataloader_clean['val'] = testing_data_loader
dataloader_clean['val_without_7'] = testing_data_loader_without_target_class


# 载入生成器
# netG_Structure = 'ResnetGenerator'
netG_Structure = 'RecursiveUnetGenerator'

# 读取参数，载入netG
if netG_Structure == 'ResnetGenerator':
    netG = ResnetGenerator(3, 3, ngf, norm_type='batch',
                           act_type='relu', gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print(
        '[bold green]ResnetGenerator netG model[/bold green] is loaded')
elif netG_Structure == 'RecursiveUnetGenerator':
    netG = RecursiveUnetGenerator(
        3, 3, num_downs=4, ngf=ngf, norm_type='batch', act_type='relu', use_dropout=True, gpu_ids=gpulist)
    netG.load_state_dict(torch.load(GENERATOR_SAVED_PATH, map_location=device))
    console.print(
        '[bold green]RecursiveUnetGenerator netG model[/bold green] is loaded:{}'.format(GENERATOR_SAVED_PATH))
else:
    raise Exception('netG loaded uncorrectly!')


# 载入干净网络模型
# Training a Resnet18  Backdoored Model
model_ft_clean = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean.fc.in_features
model_ft_clean.fc = nn.Linear(num_ftrs, 10)
model_ft_clean = model_ft_clean.to(device)
model_ft_clean.load_state_dict(torch.load(
    CLEAN_MODEL_PATH, map_location=device))
console.print(
    '[bold green]clean model[/bold green] is loaded: {}'.format(CLEAN_MODEL_PATH))


# 载入干净网络(fine_tune)模型
# Training a Resnet18  Backdoored Model
model_ft_clean_fine_tune = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_clean_fine_tune.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_clean_fine_tune.fc.in_features
model_ft_clean_fine_tune.fc = nn.Linear(num_ftrs, 10)
model_ft_clean_fine_tune = model_ft_clean_fine_tune.to(device)
model_ft_clean_fine_tune.load_state_dict(
    torch.load(CLEAN_MODEL_PATH, map_location=device))
console.print(
    '[bold green]clean model fine tune[/bold green] is loaded: {}'.format(CLEAN_MODEL_FINETUNE_PATH))

# 载入后门网络
# Training a Resnet18  Backdoored Model
model_ft_backdoor = models.resnet18()
# Finetune Final few layers to adjust for tiny imagenet input
model_ft_backdoor.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft_backdoor.fc.in_features
model_ft_backdoor.fc = nn.Linear(num_ftrs, 10)
model_ft_backdoor = model_ft_backdoor.to(device)
model_ft_backdoor.load_state_dict(torch.load(
    BACKDOOR_MODEL_PATH, map_location=device))
console.print(
    '[bold green]backdoor model[/bold green] is loaded: {}'.format(BACKDOOR_MODEL_PATH))


def entropy(_input):
    # p = softmax_func(_input)
    p = model_ft_backdoor(_input)
    p = softmax_func(p)
    # print(p)
    return (-p * p.log()).sum(1)
    # return torch.Tensor(1)

def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x)

def check(_input, _label, loader, type):
    _list = []
    console.print('\n checking..')
    # print(id(_list))
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 30:
                break
            X, Y = data
            X = X.to(device)
            _test = superimpose(_input, X, type)
            entropy_tem = entropy(_test).cpu()
            torch.cuda.empty_cache()
            _list.append(entropy_tem)
    a = torch.stack(_list).mean(0)
    # del _list
    # gc.collect()
    # return torch.stack(_list).mean(0)
    return a

def superimpose(_input1, _input2, type, alpha=None):
    if alpha is None:
        alpha = 0.5
    _input2 = _input2[:_input1.shape[0]]

    # console.print('\n input1: ',_input1.shape)

    # console.print('input2: ',_input2.shape)

    result = alpha * (_input1 - _input2) + _input2

    torchvision.utils.save_image(result,'/home/nas928/ln/GETBAK/defense_tempt_output/superimpose_{}.png'.format(type))
    torchvision.utils.save_image(transform_unNormalize(result),'/home/nas928/ln/GETBAK/defense_tempt_output/superimpose_{}_unNormalize.png'.format(type))
    return result


def detect():
    clean_entropy = []
    poison_entropy = []

    # loader = dataloader_to_detect
    # loader = track(loader, 1)

    # for i, data in enumerate(loader):
    #     _input, _label = data
    #     poison_input = add_trigger(_input)

    #     clean_entropy.append(check(_input, _label))
    #     poison_entropy.append(check(poison_input, _label))

    loader_m = dataloader_clean['val']

    for batch_index, (images, labels) in enumerate(track(loader_m,1)):
        if batch_index >20:
            break
        console.print('\nbatch_index is:{}'.format(batch_index))
        # 保存干净图片出来
        clean_images = images.clone()
        clean_images = clean_images.to(device)
        clean_images_unNomalize = transform_unNormalize(clean_images)

        images_input_netG = images
        images_input_netG = images_input_netG.to(device)

        torchvision.utils.save_image(transform_unNormalize(images_input_netG),'/home/nas928/ln/GETBAK/defense_tempt_output/images_input_netG.png')

        netG_out = netG(images_input_netG)
        netG_out, delta = normalize_and_scale(netG_out, clean_images_unNomalize)
        
        trigger_images = netG_out

        # do clamping per channel
        for cii in range(3):
            trigger_images[:,cii,:,:] = trigger_images[:,cii,:,:].clone().clamp(clean_images[:,cii,:,:].min(), clean_images[:,cii,:,:].max())


        c_e = check(clean_images, labels, loader_m, 'clean')
        p_e = check(trigger_images, labels, loader_m, 'trigger')
        
        clean_entropy.append(c_e)
        poison_entropy.append(p_e)

        console.print('\n clean_entropy:',len(clean_entropy))
        console.print('poison_entropy:',len(poison_entropy))


    clean_entropy = torch.cat(clean_entropy).flatten().sort()[0]
    poison_entropy = torch.cat(poison_entropy).flatten().sort()[0]

    _dict = {'clean': to_numpy(clean_entropy),
             'poison': to_numpy(poison_entropy)}
    result_file = './defense_strip.npy'
    np.save(result_file, _dict)

    entropy_benigh = clean_entropy.detach().numpy()
    entropy_trojan = poison_entropy.detach().numpy()

    bins = 30
    plt.figure(figsize=(10,10))
    plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
    plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
    plt.legend(loc='upper right', fontsize = 20)
    plt.ylabel('Probability (%)', fontsize = 20)
    plt.title('normalized entropy', fontsize = 20)
    plt.tick_params(labelsize=20)

    fig1 = plt.gcf()
    plt.show()
    # fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
    fig1.savefig('EntropyDNNDist_T3.png')# save the fig as pdf file

    console.print('File Saved at : ', result_file)
    console.print('Entropy Clean  Median: ', float(clean_entropy.median()))
    console.print('Entropy Poison Median: ', float(poison_entropy.median()))

    threshold_low = float(clean_entropy[int(0.05 * len(clean_entropy))])
    threshold_high = float(clean_entropy[int(0.95 * len(clean_entropy))])

    y_true = torch.cat((torch.zeros_like(clean_entropy),
                       torch.ones_like(poison_entropy)))
    entropy_t = torch.cat((clean_entropy, poison_entropy))
    y_pred = torch.where(((entropy_t < threshold_low).int() + (entropy_t > threshold_high).int()
                          ).bool(), torch.ones_like(entropy_t), torch.zeros_like(entropy_t))

    console.print(f'Threshold: ({threshold_low:5.3f}, {threshold_high:5.3f})')
    console.print("f1_score:", metrics.f1_score(y_true, y_pred))
    console.print("precision_score:", metrics.precision_score(y_true, y_pred))
    console.print("recall_score:", metrics.recall_score(y_true, y_pred))
    console.print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))

detect()